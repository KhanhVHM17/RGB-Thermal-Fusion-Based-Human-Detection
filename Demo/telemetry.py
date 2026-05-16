"""
TelemetryPoller — Real-time GPS/attitude from DJI flight records via ADB.

Flow (moi poll_interval giay):
  1. adb shell ls -t FlightRecord/ → file moi nhat
  2. adb shell stat → so sanh file size, skip neu khong doi
  3. adb pull → download file
  4. dji-log.exe --api-key → decrypt ra JSON
  5. Parse frame cuoi cung co GPS valid → update self._latest

Usage:
  poller = TelemetryPoller("dji-log.exe", "your_api_key")
  poller.start()
  ...
  data = poller.get_latest()  # {lat, lon, height, yaw, gimbal_pitch, ...}
  ...
  poller.stop()
"""

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path


# Thu tu uu tien: DJI Fly app truoc, fallback sang DJI Go v5
FLIGHT_RECORD_DIRS = [
    "/sdcard/Android/data/dji.go.v5/files/FlightRecord",               # DJI Fly (dji.go.v5) - confirmed
    "/sdcard/DJI/dji.go.v5/FlightRecord",                              # alternate path
    "/sdcard/Android/data/com.dji.industry.pilot/files/FlightRecord",  # DJI Fly Enterprise
    "/sdcard/Android/data/dji.go.v4/files/FlightRecord",               # DJI Go 4
]
FLIGHT_RECORD_DIR = FLIGHT_RECORD_DIRS[0]  # compat alias


class TelemetryPoller:
    """Poll DJI flight record via ADB, decrypt, extract latest GPS/attitude."""

    def __init__(self, dji_log_exe, api_key, adb_device=None, poll_interval=3.0):
        """
        Args:
            dji_log_exe: path to dji-log.exe
            api_key: DJI developer API key for decryption
            adb_device: optional ADB device serial (for -s flag)
            poll_interval: seconds between polls
        """
        self.dji_log_exe = str(dji_log_exe)
        self.api_key = api_key
        self.adb_device = adb_device
        self.poll_interval = poll_interval

        self._lock = threading.Lock()
        self._latest = None
        self._is_connected = False
        self._running = False
        self._thread = None
        self._last_file_size = -1
        self._last_file_path = None
        self._tmp_dir = tempfile.mkdtemp(prefix="dji_telem_")
        self._error = None

    @property
    def is_connected(self):
        with self._lock:
            return self._is_connected

    @property
    def last_error(self):
        with self._lock:
            return self._error

    def get_latest(self):
        """Return latest telemetry dict or None if not available yet."""
        with self._lock:
            return self._latest.copy() if self._latest else None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="telemetry")
        self._thread.start()
        print(f"TelemetryPoller started (interval={self.poll_interval}s)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        # Cleanup temp dir
        try:
            import shutil
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except Exception:
            pass
        print("TelemetryPoller stopped")

    # ------------------------------------------------------------------
    # ADB helpers
    # ------------------------------------------------------------------
    def _adb_cmd(self, *args):
        """Build adb command with optional device selector."""
        cmd = ["adb"]
        if self.adb_device:
            cmd.extend(["-s", self.adb_device])
        cmd.extend(args)
        return cmd

    def _run(self, cmd, timeout=10):
        """Run subprocess, return (stdout, ok). Logs stderr on failure."""
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if r.returncode != 0 and r.stderr:
                print(f"[telemetry] cmd {cmd[0]} stderr: {r.stderr.strip()[:300]}")
            return r.stdout.strip(), r.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"[telemetry] cmd {cmd[0]} timed out after {timeout}s")
            return "", False
        except FileNotFoundError:
            print(f"[telemetry] cmd not found: {cmd[0]}")
            return "not found", False

    def _check_adb(self):
        """Quick ADB connectivity check."""
        out, ok = self._run(self._adb_cmd("shell", "echo", "ok"), timeout=5)
        return ok and "ok" in out

    # ------------------------------------------------------------------
    # Flight record discovery
    # ------------------------------------------------------------------
    def _find_latest_record(self):
        """Find the most recent FlightRecord file, thu toan bo app paths."""
        for d in FLIGHT_RECORD_DIRS:
            # -p: them trailing '/' vao thu muc, de loc ra khoi danh sach
            out, ok = self._run(self._adb_cmd("shell", "ls", "-pt", d + "/"))
            if not ok or not out:
                continue
            files = [f.strip() for f in out.splitlines()
                     if f.strip()
                     and not f.strip().startswith("ls:")
                     and not f.strip().endswith("/")]   # bo qua thu muc (si_cache, ...)
            if files:
                print(f"TelemetryPoller: found FlightRecord in {d}: {files[0]}")
                return f"{d}/{files[0]}"
        return None

    def _get_file_size(self, remote_path):
        """Get file size on device. Returns -1 on failure."""
        # wc -c is more portable than stat -c on Android
        out, ok = self._run(self._adb_cmd("shell", "wc", "-c", remote_path))
        if not ok or not out:
            return -1
        try:
            return int(out.split()[0])
        except (ValueError, IndexError):
            return -1

    # ------------------------------------------------------------------
    # Decrypt + parse
    # ------------------------------------------------------------------
    def _pull_and_decrypt(self, remote_path):
        """Pull file from device, decrypt with dji-log.exe, return parsed frames."""
        local_log = os.path.join(self._tmp_dir, "current_flight.txt")
        local_json = os.path.join(self._tmp_dir, "frames.json")

        # Pull
        pull_out, ok = self._run(self._adb_cmd("pull", remote_path, local_log), timeout=15)
        if not ok:
            print(f"[telemetry] adb pull failed: {pull_out}")
            return None
        print(f"[telemetry] pulled {remote_path} -> {local_log}")

        # Decrypt — remove stale output file first
        if os.path.exists(local_json):
            os.remove(local_json)

        cmd = [
            self.dji_log_exe,
            "--api-key", self.api_key,
            "-o", local_json,
            local_log,
        ]
        stdout, ok = self._run(cmd, timeout=30)
        if stdout:
            print(f"[telemetry] dji-log stdout: {stdout[:200]}")

        if not os.path.exists(local_json):
            # Fallback: dji-log may write to stdout instead of file
            if stdout and stdout.lstrip().startswith("{"):
                print("[telemetry] dji-log wrote JSON to stdout, using that")
                try:
                    data = json.loads(stdout)
                    frames = data.get("frames", data if isinstance(data, list) else [])
                    print(f"[telemetry] decoded {len(frames)} frames from stdout")
                    return frames
                except json.JSONDecodeError as e:
                    print(f"[telemetry] JSON decode error from stdout: {e}")
            print(f"[telemetry] decrypt failed: output file not created (ok={ok})")
            return None

        # Parse JSON file
        try:
            with open(local_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Handle both {"frames": [...]} and direct list
            if isinstance(data, list):
                frames = data
            else:
                frames = data.get("frames", [])
                if not frames:
                    # Log top-level keys to help debug structure
                    print(f"[telemetry] JSON top-level keys: {list(data.keys())[:10]}")
            print(f"[telemetry] decoded {len(frames)} frames")
            return frames
        except (json.JSONDecodeError, OSError) as e:
            print(f"[telemetry] JSON parse error: {e}")
            return None

    def _extract_latest_gps(self, frames):
        """Find the last frame with valid GPS (lat != 0)."""
        if not frames:
            return None

        # Debug: log last frame structure once
        last = frames[-1]
        osd_keys = list(last.get("osd", {}).keys())[:15]
        frame_keys = list(last.keys())
        print(f"[telemetry] last frame keys: {frame_keys}, osd keys: {osd_keys}")

        for frame in reversed(frames):
            osd = frame.get("osd", {})
            lat = osd.get("latitude", 0)
            lon = osd.get("longitude", 0)
            if abs(lat) > 0.01 and abs(lon) > 0.01:
                gimbal = frame.get("gimbal", {})
                return {
                    "lat": lat,
                    "lon": lon,
                    "altitude": osd.get("altitude", 0),
                    "height": osd.get("height", 0),
                    "yaw": osd.get("yaw", 0),
                    "pitch": osd.get("pitch", 0),
                    "roll": osd.get("roll", 0),
                    "gimbal_pitch": gimbal.get("pitch", -90),
                    "gimbal_yaw": gimbal.get("yaw", 0),
                    "gimbal_roll": gimbal.get("roll", 0),
                    "gpsNum": osd.get("gpsNum", 0),
                    "flyTime": osd.get("flyTime", 0),
                    "flycState": osd.get("flycState", "Unknown"),
                    "timestamp": time.time(),
                }

        # No valid GPS found — log sample values for diagnosis
        if frames:
            sample = frames[-1].get("osd", {})
            home  = frames[-1].get("home", {})
            print(f"[telemetry] no valid GPS in {len(frames)} frames. "
                  f"lat={sample.get('latitude', 'N/A')} lon={sample.get('longitude', 'N/A')} "
                  f"height={sample.get('height', 'N/A')}m gpsNum={sample.get('gpsNum', 'N/A')} "
                  f"home_lat={home.get('latitude', 'N/A')} home_lon={home.get('longitude', 'N/A')}")

        # Fallback: try home point GPS (recorded when drone armed)
        for frame in reversed(frames):
            home = frame.get("home", {})
            lat = home.get("latitude", 0)
            lon = home.get("longitude", 0)
            if abs(lat) > 0.01 and abs(lon) > 0.01:
                osd = frame.get("osd", {})
                gimbal = frame.get("gimbal", {})
                print(f"[telemetry] using home-point GPS fallback: lat={lat:.6f} lon={lon:.6f}")
                return {
                    "lat": lat,
                    "lon": lon,
                    "altitude": osd.get("altitude", 0),
                    "height": osd.get("height", 0),
                    "yaw": osd.get("yaw", 0),
                    "pitch": osd.get("pitch", 0),
                    "roll": osd.get("roll", 0),
                    "gimbal_pitch": gimbal.get("pitch", -90),
                    "gimbal_yaw": gimbal.get("yaw", 0),
                    "gimbal_roll": gimbal.get("roll", 0),
                    "gpsNum": osd.get("gpsNum", 0),
                    "flyTime": osd.get("flyTime", 0),
                    "flycState": osd.get("flycState", "Unknown"),
                    "timestamp": time.time(),
                    "gps_source": "home_point",
                }
        return None

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------
    def _poll_loop(self):
        while self._running:
            try:
                self._poll_once()
            except Exception as e:
                with self._lock:
                    self._error = str(e)
                    self._is_connected = False
            time.sleep(self.poll_interval)

    def _poll_once(self):
        # Check ADB
        if not self._check_adb():
            with self._lock:
                self._is_connected = False
                self._error = "ADB not connected"
            return

        with self._lock:
            self._is_connected = True

        # Find latest flight record
        remote_path = self._find_latest_record()
        if not remote_path:
            with self._lock:
                self._error = "No FlightRecord found on device"
            return

        # Check file size — skip if unchanged
        size = self._get_file_size(remote_path)
        if size == self._last_file_size and remote_path == self._last_file_path:
            return  # File unchanged, skip expensive decrypt
        self._last_file_size = size
        self._last_file_path = remote_path

        # Pull + decrypt + parse
        frames = self._pull_and_decrypt(remote_path)
        if not frames:
            with self._lock:
                self._error = "Failed to decrypt flight record"
            return

        telem = self._extract_latest_gps(frames)
        if telem:
            with self._lock:
                self._latest = telem
                self._error = None


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    exe = sys.argv[1] if len(sys.argv) > 1 else "dji-log.exe"
    key = sys.argv[2] if len(sys.argv) > 2 else "cb501f609e7d2d46b6ab0252938336e"

    print(f"Testing TelemetryPoller with {exe}")
    poller = TelemetryPoller(exe, key, poll_interval=3.0)
    poller.start()

    try:
        for i in range(10):
            time.sleep(3)
            data = poller.get_latest()
            connected = poller.is_connected
            err = poller.last_error
            print(f"[{i+1}] connected={connected} error={err}")
            if data:
                print(f"  lat={data['lat']:.6f} lon={data['lon']:.6f} "
                      f"height={data['height']:.1f}m yaw={data['yaw']:.1f} "
                      f"gimbal_pitch={data['gimbal_pitch']:.1f} "
                      f"gps_sats={data['gpsNum']} state={data['flycState']}")
    except KeyboardInterrupt:
        pass
    finally:
        poller.stop()
