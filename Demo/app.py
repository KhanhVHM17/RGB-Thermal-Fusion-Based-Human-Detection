"""
FastAPI Demo App -- RGB-Thermal Drone Person Detection + GPS
Architecture: Producer-Consumer (3+ threads)
  - Capture thread: doc frame tu video (hoac screen capture live), day vao queue
  - Inference thread: lay frame tu queue, chay model, luu ket qua
  - Stream (generate_frames): lay frame + detections da cached, encode MJPEG
  - Telemetry thread (live mode): poll DJI flight record qua ADB, update drone GPS

Chay: cd Demo && uvicorn app:app --host 0.0.0.0 --port 8000
"""

import io
import os
import time
import json
import threading
from queue import Queue, Empty
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Lazy import -- inference.py needs torch which may not be on local dev machine
FusionDetector = None

def _import_detector():
    global FusionDetector
    if FusionDetector is None:
        from inference import FusionDetector as _FD
        FusionDetector = _FD

# =====================================================================
# CONFIG
# =====================================================================
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Model config -- thay doi tuy theo model da train
MODEL_MODE = os.environ.get("MODEL_MODE", "onnx")  # "onnx", "mid", or "yolo"
MODEL_ARCH = os.environ.get("MODEL_ARCH", "progressive")  # "baseline" or "progressive"
MODEL_PATH = os.environ.get("MODEL_PATH", str(MODELS_DIR / "fusion_progressive_finetune.onnx"))
RGB_BB_PATH = os.environ.get("RGB_BB_PATH", str(MODELS_DIR / "llvip_rgb_best.pt"))
THR_BB_PATH = os.environ.get("THR_BB_PATH", str(MODELS_DIR / "llvip_thermal_best.pt"))

CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.1"))
STREAM_WIDTH = int(os.environ.get("STREAM_WIDTH", "1280"))
FRAME_QUEUE_SIZE = int(os.environ.get("FRAME_QUEUE_SIZE", "2"))

# Detection history cap -- cho heat map coverage
HISTORY_MAX = 5000

# Live mode config
DJI_API_KEY = os.environ.get("DJI_API_KEY", "cb501f609e7d2d46b6ab0252938336e")
DJI_LOG_EXE = os.environ.get("DJI_LOG_EXE", str(BASE_DIR / "dji-log.exe"))

# =====================================================================
# APP
# =====================================================================
app = FastAPI(title="Drone Person Detection Demo")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Global state
detector = None
video_path: Optional[str] = None
thermal_video_path: Optional[str] = None
is_running: bool = False
is_paused: bool = False
seek_target: Optional[float] = None
current_session: str = ""  # session id per video upload

# Shared state (thread-safe)
state_lock = threading.Lock()
current_detections: list = []
video_fps: float = 0.0
inference_fps: float = 0.0
video_total_frames: int = 0
video_current_frame: int = 0

# Detection history -- tich luy cho heat map coverage
detection_history: list = []

# Detection saves -- luu lich su detect: crop anh + GPS (cho History tab)
detection_saves: list = []   # [{id, timestamp, conf, lat, lon, bbox, img_bytes}]
SAVES_MAX = 100              # toi da 100 anh
_last_save_time: float = 0.0 # cooldown: max 1 save / 3 giay

# Flagged detections -- co cam persistent (khong xoa moi frame)
flagged_detections: list = []

# Detection timeline -- so nguoi detect theo thoi gian (cho dashboard)
detection_timeline: list = []  # [{ts, count, frame}]
TIMELINE_MAX = 1000

# Latest annotated frame (thread-safe) cho MJPEG stream
frame_lock = threading.Lock()
latest_frame: Optional[np.ndarray] = None
latest_raw_frame: Optional[np.ndarray] = None  # full resolution, chua annotate

# Queues
capture_queue: Queue = Queue(maxsize=FRAME_QUEUE_SIZE)
adb_preview_queue: Queue = Queue(maxsize=2)  # Raw phone screen cho preview

# Thread handles
capture_thread: Optional[threading.Thread] = None
inference_thread: Optional[threading.Thread] = None

# Live mode state
live_mode: bool = False
live_thread: Optional[threading.Thread] = None
telemetry_poller = None  # TelemetryPoller instance

# ADB preview-only mode (chay doc lap voi live_mode)
adb_preview_mode: bool = False


def load_model():
    global detector
    if detector is not None:
        return
    try:
        _import_detector()
        if FusionDetector is None:
            print("Warning: torch not available, running without detection")
            return
        if MODEL_MODE == "onnx":
            detector = FusionDetector(MODEL_PATH, mode="onnx")
        elif MODEL_MODE == "mid":
            detector = FusionDetector(MODEL_PATH, RGB_BB_PATH, THR_BB_PATH,
                                      mode="mid", arch=MODEL_ARCH)
        else:
            detector = FusionDetector(MODEL_PATH, mode="yolo")
    except Exception as e:
        print(f"Model load failed: {e}")
        print("Demo will run without detection. Upload a valid model to Demo/models/")


# =====================================================================
# CAPTURE THREAD -- doc frame tu video, day vao queue
# =====================================================================
def capture_worker():
    global video_total_frames, video_current_frame, is_running, latest_frame

    while True:
        if video_path is None or not is_running:
            time.sleep(0.1)
            continue

        cap_rgb = cv2.VideoCapture(video_path)
        if not cap_rgb.isOpened():
            time.sleep(1)
            continue

        # Thermal video (optional — neu khong co thi dung RGB lam thermal)
        cap_thr = None
        if thermal_video_path:
            cap_thr = cv2.VideoCapture(thermal_video_path)
            if not cap_thr.isOpened():
                print(f"Warning: cannot open thermal video {thermal_video_path}")
                cap_thr = None

        src_fps = cap_rgb.get(cv2.CAP_PROP_FPS) or 25.0
        target_fps = min(src_fps, 30.0)
        frame_delay = 1.0 / target_fps
        frame_skip = max(1, int(round(src_fps / target_fps)))

        with state_lock:
            video_total_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        while is_running and cap_rgb.isOpened():
            # Seek
            global seek_target
            if seek_target is not None:
                target_frame = int(seek_target * video_total_frames)
                cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                if cap_thr:
                    cap_thr.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                seek_target = None

            # Pause
            if is_paused:
                time.sleep(0.05)
                continue

            t0 = time.time()

            # Skip frames neu video goc fps > 30
            for _ in range(frame_skip - 1):
                cap_rgb.grab()
                if cap_thr:
                    cap_thr.grab()

            ret_rgb, frame_rgb = cap_rgb.read()
            if not ret_rgb:
                cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if cap_thr:
                    cap_thr.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Doc thermal frame (dong bo voi RGB)
            frame_thr = None
            if cap_thr:
                ret_thr, frame_thr = cap_thr.read()
                if not ret_thr:
                    cap_thr.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    _, frame_thr = cap_thr.read()

            with state_lock:
                video_current_frame = int(cap_rgb.get(cv2.CAP_PROP_POS_FRAMES))

            # Day (rgb, thermal) vao queue
            pair = (frame_rgb, frame_thr)
            if not capture_queue.full():
                capture_queue.put(pair)
            else:
                try:
                    capture_queue.get_nowait()
                except Empty:
                    pass
                capture_queue.put(pair)

            # Frame rate control
            elapsed = time.time() - t0
            wait = max(0, frame_delay - elapsed)
            if wait > 0:
                time.sleep(wait)

        cap_rgb.release()
        if cap_thr:
            cap_thr.release()


# =====================================================================
# LIVE CAPTURE THREAD -- screen capture via mss, auto grayscale thermal
# =====================================================================
def live_capture_worker(monitor_idx, crop_region):
    """Capture screen region, generate pseudo-thermal, push to queue.

    Args:
        monitor_idx: mss monitor index (1 = primary screen)
        crop_region: dict {left, top, width, height} in pixels
    """
    global live_mode

    try:
        import mss
    except ImportError:
        print("ERROR: mss not installed. Run: pip install mss")
        live_mode = False
        return

    target_fps = 30
    frame_delay = 1.0 / target_fps

    with mss.mss() as sct:
        region = {
            "left": crop_region["left"],
            "top": crop_region["top"],
            "width": crop_region["width"],
            "height": crop_region["height"],
        }
        print(f"Live capture started: monitor={monitor_idx} region={region}")

        while live_mode:
            t0 = time.time()

            # Capture screen region (BGRA)
            screenshot = sct.grab(region)
            frame_rgb = np.array(screenshot)[:, :, :3].copy()  # BGRA -> BGR

            # Pseudo-thermal: grayscale -> COLORMAP_HOT (bright=warm, simulates IR)
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
            frame_thr = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)

            # Push (rgb, thermal) pair to queue
            pair = (frame_rgb, frame_thr)
            if not capture_queue.full():
                capture_queue.put(pair)
            else:
                try:
                    capture_queue.get_nowait()
                except Empty:
                    pass
                capture_queue.put(pair)

            # Auto-update drone config from telemetry
            if telemetry_poller and detector:
                telem = telemetry_poller.get_latest()
                if telem:
                    detector.update_drone_config(
                        lat=telem['lat'],
                        lon=telem['lon'],
                        alt=telem['height'],  # AGL height
                        yaw=telem['yaw'],
                        pitch=telem['gimbal_pitch'],  # Camera pitch from gimbal
                    )

            # Frame rate control
            elapsed = time.time() - t0
            wait = max(0, frame_delay - elapsed)
            if wait > 0:
                time.sleep(wait)

    print("Live capture stopped.")


# =====================================================================
# INFERENCE THREAD -- lay frame, chay model, update shared state
# =====================================================================
def inference_worker():
    global current_detections, inference_fps, latest_frame, latest_raw_frame

    while True:
        # Drain queue de luon lay frame MOI NHAT, bo qua frame cu neu co nhieu frame cho
        pair = None
        while True:
            try:
                pair = capture_queue.get_nowait()
            except Empty:
                break
        if pair is None:
            try:
                pair = capture_queue.get(timeout=0.5)
            except Empty:
                continue

        # Unpack (rgb_frame, thermal_frame) pair
        if isinstance(pair, tuple):
            frame, thr_frame = pair
        else:
            frame, thr_frame = pair, None  # backward compat

        t0 = time.time()

        dets = []
        if detector is not None:
            try:
                dets = detector.detect_frame(frame, thr_frame=thr_frame, conf=CONF_THRESHOLD)
            except Exception as e:
                print(f"Detection error: {e}")

        # Luu raw frame cho snapshot (truoc khi ve bbox)
        with frame_lock:
            latest_raw_frame = frame.copy()

        # Luu detection saves (history) -- cooldown 3 giay
        global _last_save_time
        if dets:
            now = time.time()
            if now - _last_save_time >= 3.0:
                _last_save_time = now
                best = max(dets, key=lambda d: d['conf'])
                x1, y1, x2, y2 = best['bbox']
                fh, fw = frame.shape[:2]
                pad = max(10, int((x2 - x1) * 0.2))
                sx1, sy1 = max(0, x1 - pad), max(0, y1 - pad)
                sx2, sy2 = min(fw, x2 + pad), min(fh, y2 + pad)
                crop = frame[sy1:sy2, sx1:sx2]
                if crop.size > 0:
                    _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    with state_lock:
                        save_id = len(detection_saves) + 1
                        detection_saves.append({
                            'id': save_id,
                            'timestamp': now,
                            'conf': best['conf'],
                            'lat': best.get('lat'),
                            'lon': best.get('lon'),
                            'bbox': best['bbox'],
                            'img': buf.tobytes(),
                        })
                        if len(detection_saves) > SAVES_MAX:
                            detection_saves.pop(0)

        # Draw detections
        annotated = frame
        if detector is not None and dets:
            annotated = detector.draw_detections(frame, dets)

        # Info overlay
        with state_lock:
            fps_display = video_fps
        cv2.putText(annotated, f"FPS: {fps_display:.1f}  |  {len(dets)} person(s)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Resize de giam bandwidth MJPEG
        h, w = annotated.shape[:2]
        if w > STREAM_WIDTH:
            scale = STREAM_WIDTH / w
            annotated = cv2.resize(annotated, (STREAM_WIDTH, int(h * scale)))

        # Update shared state
        elapsed = time.time() - t0
        with state_lock:
            current_detections = dets
            inference_fps = 1.0 / max(elapsed, 1e-6)
            # Tich luy detection history cho heat map
            for d in dets:
                if d.get('lat') is not None:
                    detection_history.append({
                        'lat': d['lat'], 'lon': d['lon'],
                        'conf': d['conf'], 'frame': video_current_frame
                    })
            if len(detection_history) > HISTORY_MAX:
                del detection_history[:len(detection_history) - HISTORY_MAX]
            # Timeline — so nguoi detect theo thoi gian (dashboard)
            detection_timeline.append({
                'ts': time.time(),
                'count': len(dets),
                'frame': video_current_frame,
            })
            if len(detection_timeline) > TIMELINE_MAX:
                del detection_timeline[:len(detection_timeline) - TIMELINE_MAX]

        with frame_lock:
            latest_frame = annotated


# =====================================================================
# ENDPOINTS
# =====================================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...),
                       thermal_file: UploadFile = File(...)):
    global video_path, thermal_video_path, is_running, is_paused, latest_frame, current_session
    is_running = False
    time.sleep(0.3)

    # Save RGB video
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Save Thermal video (bat buoc)
    thr_save = UPLOAD_DIR / f"thermal_{thermal_file.filename}"
    with open(thr_save, "wb") as f:
        content = await thermal_file.read()
        f.write(content)
    thermal_video_path = str(thr_save)

    # Clear queue
    while not capture_queue.empty():
        try:
            capture_queue.get_nowait()
        except Empty:
            break

    # New session — clear flags + history + timeline
    current_session = f"{file.filename}_{int(time.time())}"
    with state_lock:
        flagged_detections.clear()
        detection_history.clear()
        detection_timeline.clear()
        detection_saves.clear()

    with frame_lock:
        latest_frame = None

    video_path = str(save_path)
    is_paused = False
    is_running = True
    return {"status": "ok", "filename": file.filename,
            "thermal_filename": thermal_file.filename,
            "path": video_path, "session": current_session}


class DroneConfig(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt: Optional[float] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    hfov: Optional[float] = None
    vfov: Optional[float] = None


@app.post("/drone_config")
async def update_drone_config(config: DroneConfig):
    if detector is None:
        return {"status": "error", "msg": "model not loaded"}
    updates = {k: v for k, v in config.model_dump().items() if v is not None}
    detector.update_drone_config(**updates)
    return {"status": "ok", "drone_config": detector.drone_config}


@app.post("/set_conf")
async def set_conf(request: Request):
    global CONF_THRESHOLD
    body = await request.json()
    val = float(body.get("conf", CONF_THRESHOLD))
    CONF_THRESHOLD = max(0.05, min(0.95, val))
    return {"status": "ok", "conf": CONF_THRESHOLD}


@app.get("/detection_saves")
async def get_detection_saves():
    """Tra ve danh sach detection saves (khong co img bytes)."""
    with state_lock:
        items = [
            {k: v for k, v in s.items() if k != 'img'}
            for s in detection_saves
        ]
    return JSONResponse(content={"saves": items, "total": len(items)})


@app.get("/detection_save_img/{save_id}")
async def get_detection_save_img(save_id: int):
    """Tra ve JPEG crop cua detection save theo id."""
    with state_lock:
        entry = next((s for s in detection_saves if s['id'] == save_id), None)
    if entry is None:
        from fastapi.responses import Response
        return Response(status_code=404)
    return StreamingResponse(io.BytesIO(entry['img']), media_type="image/jpeg")


@app.delete("/detection_saves")
async def clear_detection_saves():
    with state_lock:
        detection_saves.clear()
    global _last_save_time
    _last_save_time = 0.0
    return {"status": "ok"}


@app.post("/pause")
async def pause_video():
    global is_paused
    is_paused = True
    return {"status": "ok", "paused": True}


@app.post("/resume")
async def resume_video():
    global is_paused
    is_paused = False
    return {"status": "ok", "paused": False}


@app.post("/seek")
async def seek_video(request: Request):
    global seek_target
    body = await request.json()
    ratio = float(body.get("ratio", 0))
    seek_target = max(0.0, min(1.0, ratio))
    return {"status": "ok", "ratio": seek_target}


@app.get("/detections")
async def get_detections():
    # Tinh GSD tu drone config
    gsd = None
    if detector is not None:
        try:
            from inference import compute_gsd
            dc = detector.drone_config
            gsd = compute_gsd(dc['alt'], dc['hfov'], dc['vfov'], 640, 640)
        except Exception:
            pass
    with state_lock:
        progress = 0.0
        if video_total_frames > 0:
            progress = video_current_frame / video_total_frames
        return JSONResponse(content={
            "detections": current_detections,
            "fps": round(inference_fps, 1),
            "gsd": gsd,
            "video_loaded": video_path is not None,
            "model_loaded": detector is not None,
            "paused": is_paused,
            "progress": round(progress, 4),
            "current_frame": video_current_frame,
            "total_frames": video_total_frames,
        })


@app.get("/status")
async def status():
    return {
        "model_loaded": detector is not None,
        "model_mode": MODEL_MODE,
        "video_loaded": video_path is not None,
        "video_path": video_path,
        "thermal_loaded": thermal_video_path is not None,
        "is_running": is_running,
        "live_mode": live_mode,
        "fps": round(inference_fps, 1),
        "drone_config": detector.drone_config if detector else None,
    }


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/detection_history")
async def get_detection_history():
    with state_lock:
        return JSONResponse(content={
            "history": detection_history,
            "count": len(detection_history)
        })


@app.get("/detection_snapshot")
async def detection_snapshot(x1: int, y1: int, x2: int, y2: int):
    """Crop bbox tu latest raw frame, pad 60% context, upscale len 192px, tra ve JPEG."""
    with frame_lock:
        frame = latest_raw_frame
    if frame is None:
        return JSONResponse(content={"error": "no frame"}, status_code=404)
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    # Pad 60% moi phia de lay them context xung quanh nguoi
    pad_x = int(bw * 0.6)
    pad_y = int(bh * 0.6)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    if cx2 <= cx1 or cy2 <= cy1:
        return JSONResponse(content={"error": "invalid bbox"}, status_code=400)
    crop = frame[cy1:cy2, cx1:cx2]
    # Upscale len toi thieu 192px (chieu nho nhat) de thumbnail ro hon
    ch, cw = crop.shape[:2]
    min_side = min(ch, cw)
    if min_side < 192:
        scale = 192 / min_side
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_LANCZOS4)
    _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


@app.get("/flagged_detections")
async def get_flagged():
    with state_lock:
        return JSONResponse(content={"flags": flagged_detections})


@app.delete("/flagged_detections")
async def clear_flags():
    with state_lock:
        flagged_detections.clear()
    return {"status": "ok", "total_flags": 0}


@app.post("/flag_detection")
async def flag_detection(request: Request):
    body = await request.json()
    with state_lock:
        flagged_detections.append({
            "lat": body["lat"], "lon": body["lon"],
            "conf": body["conf"], "id": body["id"],
            "bbox": body.get("bbox"),
            "frame": video_current_frame,
            "timestamp": time.time(),
        })
    return {"status": "ok", "total_flags": len(flagged_detections)}


@app.delete("/flag_detection/{flag_id}")
async def delete_flag(flag_id: int):
    """Xoa 1 flag theo id."""
    with state_lock:
        before = len(flagged_detections)
        flagged_detections[:] = [f for f in flagged_detections if f.get("id") != flag_id]
        removed = before - len(flagged_detections)
    return {"status": "ok", "removed": removed, "total_flags": len(flagged_detections)}


@app.get("/detection_timeline")
async def get_timeline():
    """Timeline so nguoi detect theo thoi gian — cho dashboard chart."""
    with state_lock:
        return JSONResponse(content={
            "timeline": detection_timeline,
            "count": len(detection_timeline),
        })


# =====================================================================
# LIVE MODE ENDPOINTS
# =====================================================================
def _find_windows(keyword="scrcpy"):
    """Find windows matching keyword using Windows API (ctypes). No extra deps."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    EnumWindows = user32.EnumWindows
    GetWindowTextW = user32.GetWindowTextW
    GetWindowTextLengthW = user32.GetWindowTextLengthW
    IsWindowVisible = user32.IsWindowVisible
    GetWindowRect = user32.GetWindowRect

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    results = []

    def callback(hwnd, _):
        if not IsWindowVisible(hwnd):
            return True
        length = GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if keyword.lower() in title.lower():
            rect = wintypes.RECT()
            GetWindowRect(hwnd, ctypes.byref(rect))
            results.append({
                "title": title,
                "hwnd": int(hwnd),
                "x": rect.left,
                "y": rect.top,
                "w": rect.right - rect.left,
                "h": rect.bottom - rect.top,
            })
        return True

    EnumWindows(WNDENUMPROC(callback), 0)
    return results


def _get_client_rect_screen(hwnd):
    """Lay client area (phan noi dung, khong co title bar/border) theo toa do man hinh."""
    import ctypes
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
    user32 = ctypes.windll.user32
    cr = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(cr)):
        return None
    pt = POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return None
    w = cr.right - cr.left
    h = cr.bottom - cr.top
    if w <= 0 or h <= 0:
        return None
    return pt.x, pt.y, w, h


def _capture_window_by_mss(hwnd):
    """Capture CLIENT area cua window (khong co title bar) dung mss.
    Window phai visible (khong bi minimize). Tra ve BGR numpy array hoac None.
    """
    rect = _get_client_rect_screen(hwnd)
    if rect is None:
        return None
    x, y, w, h = rect
    try:
        import mss
        with mss.mss() as sct:
            region = {"left": x, "top": y, "width": w, "height": h}
            shot = sct.grab(region)
            frame = np.array(shot)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    except Exception:
        return None




def scrcpy_preview_only_worker(hwnd: int):
    """Preview-only Win32 capture: feed adb_preview_queue, dung khi live_mode=True."""
    global adb_preview_mode
    print(f"Scrcpy preview worker started (hwnd={hwnd})")
    while adb_preview_mode:
        if live_mode:
            time.sleep(0.1)
            continue
        frame = _capture_window_by_mss(hwnd)
        if frame is not None:
            _push_preview_frame(frame)
        time.sleep(0.067)  # ~15fps
    print("Scrcpy preview worker stopped.")


def live_capture_worker_hwnd(hwnd, width, height):
    """Capture scrcpy window by HWND -- works with window in background."""
    global live_mode

    target_fps = 30
    frame_delay = 1.0 / target_fps

    print(f"Live capture (HWND={hwnd}) started: {width}x{height}")

    while live_mode:
        t0 = time.time()

        frame_rgb = _capture_window_by_mss(hwnd)
        if frame_rgb is None:
            time.sleep(0.1)
            continue

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        frame_thr = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)

        pair = (frame_rgb, frame_thr)
        if not capture_queue.full():
            capture_queue.put(pair)
        else:
            try:
                capture_queue.get_nowait()
            except Empty:
                pass
            capture_queue.put(pair)

        if telemetry_poller and detector:
            telem = telemetry_poller.get_latest()
            if telem:
                detector.update_drone_config(
                    lat=telem['lat'],
                    lon=telem['lon'],
                    alt=telem['height'],
                    yaw=telem['yaw'],
                    pitch=telem['gimbal_pitch'],
                )

        elapsed = time.time() - t0
        wait = max(0, frame_delay - elapsed)
        if wait > 0:
            time.sleep(wait)

    print("Live capture (HWND) stopped.")


class AdbScreenStream:
    """Persistent ADB screen stream: 1 process chay loop, parse PNG tu stdout.
    Loai bo overhead spawn process moi frame (~100ms tren Windows).
    """
    PNG_MAGIC = b'\x89PNG'
    PNG_IEND  = b'IEND\xaeB`\x82'  # 4 bytes IEND + 4 bytes CRC (luon co dinh)

    def __init__(self, adb_base: list):
        import subprocess as _sp
        # sleep 0.033 tren phone: cap ~30fps
        cmd = adb_base + ["exec-out", "sh", "-c",
                          "while true; do screencap -p; sleep 0.033; done"]
        self._proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.DEVNULL)
        self._buf   = bytearray()   # bytearray: extend() khong copy toan bo vung nho
        self._frame = None
        self._lock  = threading.Lock()
        self._alive = True
        t = threading.Thread(target=self._read_loop, daemon=True, name="adb_stream")
        t.start()

    def _read_loop(self):
        CHUNK = 65536
        while self._alive:
            try:
                chunk = self._proc.stdout.read(CHUNK)
            except Exception:
                break
            if not chunk:
                break
            self._buf.extend(chunk)   # extend: append in-place, khong tao copy
            # Parse tat ca cac PNG frame hoan chinh trong buffer
            while True:
                s = self._buf.find(self.PNG_MAGIC)
                if s == -1:
                    self._buf.clear()
                    break
                e = self._buf.find(self.PNG_IEND, s)
                if e == -1:
                    del self._buf[:s]   # xoa phan truoc PNG start, khong copy
                    break
                e += len(self.PNG_IEND)
                # Copy PNG frame ra truoc, sau do moi resize buffer
                png_bytes = bytes(self._buf[s:e])
                del self._buf[:e]
                arr = np.frombuffer(png_bytes, dtype=np.uint8)
                f = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if f is not None:
                    with self._lock:
                        self._frame = f

    def get_frame(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._alive = False
        try:
            self._proc.kill()
        except Exception:
            pass


def _push_preview_frame(frame):
    """Resize va day frame vao adb_preview_queue (replace neu full)."""
    ph, pw = frame.shape[:2]
    if pw > 854:
        preview = cv2.resize(frame, (854, int(ph * 854 / pw)))
    else:
        preview = frame
    _, jpg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if adb_preview_queue.full():
        try:
            adb_preview_queue.get_nowait()
        except Empty:
            pass
    adb_preview_queue.put(jpg.tobytes())


def adb_preview_only_worker(device_serial: str):
    """Preview-only ADB capture dung AdbScreenStream (persistent subprocess)."""
    global adb_preview_mode
    adb_base = ["adb", "-s", device_serial] if device_serial else ["adb"]
    print(f"ADB preview worker started (device={device_serial})")
    stream = AdbScreenStream(adb_base)
    try:
        while adb_preview_mode:
            if live_mode:
                time.sleep(0.1)
                continue
            frame = stream.get_frame()
            if frame is not None:
                _push_preview_frame(frame)
            time.sleep(0.08)  # ~12fps cap
    finally:
        stream.stop()
    print("ADB preview worker stopped.")


def adb_screen_capture_worker(device_serial: str = None, crop: dict = None):
    """Capture phone screen via ADB screencap dung AdbScreenStream (persistent subprocess).
    - Feeds inference queue (capture_queue) AND preview queue (adb_preview_queue)
    - crop: {x, y, w, h} trong pixel man hinh dien thoai, None = full screen
    """
    global live_mode

    adb_base = ["adb", "-s", device_serial] if device_serial else ["adb"]
    print(f"ADB screen capture started (device={device_serial or 'default'}, crop={crop})")
    stream = AdbScreenStream(adb_base)
    try:
        while live_mode:
            t0 = time.time()
            try:
                frame = stream.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                # Apply crop truoc khi inference
                if crop:
                    x, y, w, h = int(crop["x"]), int(crop["y"]), int(crop["w"]), int(crop["h"])
                    fh, fw = frame.shape[:2]
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(fw, x + w), min(fh, y + h)
                    if x2 > x1 and y2 > y1:
                        frame = frame[y1:y2, x1:x2]

                # Preview
                _push_preview_frame(frame)

                # Inference: pseudo-thermal = COLORMAP_HOT (dong nhat voi training)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                pair = (frame, cv2.applyColorMap(gray, cv2.COLORMAP_HOT))
                if capture_queue.full():
                    try:
                        capture_queue.get_nowait()
                    except Empty:
                        pass
                capture_queue.put(pair)

                # Sync telemetry
                if telemetry_poller and detector:
                    telem = telemetry_poller.get_latest()
                    if telem:
                        detector.update_drone_config(
                            lat=telem["lat"], lon=telem["lon"],
                            alt=telem["height"], yaw=telem["yaw"],
                            pitch=telem["gimbal_pitch"],
                        )

            except Exception as exc:
                print(f"ADB screencap error: {exc}")
                time.sleep(0.5)

            elapsed = time.time() - t0
            if elapsed < 0.033:
                time.sleep(0.033 - elapsed)  # cap ~30fps
    finally:
        stream.stop()
    print("ADB screen capture stopped.")


def generate_adb_preview():
    """MJPEG generator: stream raw phone screen cho preview trong web."""
    placeholder = None
    while True:
        try:
            jpg = adb_preview_queue.get(timeout=2.0)
        except Empty:
            # Neu queue rong (live mode chua bat hoac dung), gui placeholder
            if placeholder is None:
                img = np.zeros((240, 480, 3), dtype=np.uint8)
                cv2.putText(img, "ADB preview", (140, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                _, enc = cv2.imencode(".jpg", img)
                placeholder = enc.tobytes()
            jpg = placeholder

        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + jpg
            + b"\r\n"
        )


_scrcpy_proc = None  # global handle de tranh zombie process

@app.post("/start_scrcpy")
async def start_scrcpy_endpoint(request: Request):
    """Khoi dong scrcpy o background (no-audio, max-fps 30)."""
    global _scrcpy_proc
    import subprocess as _sp
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    device_serial = (body.get("device_serial") or "").strip()
    # Neu process van chay nhung window da mat -> kill va restart
    if _scrcpy_proc and _scrcpy_proc.poll() is None:
        window_still_alive = bool(_find_windows("scrcpy-drone") or _find_windows("scrcpy"))
        if window_still_alive:
            return JSONResponse(content={"ok": True, "msg": "already running"})
        # Window mat -> kill process zombie va restart
        try:
            _scrcpy_proc.kill()
        except Exception:
            pass
        _scrcpy_proc = None
    # Tim scrcpy theo thu tu uu tien
    candidates = [
        BASE_DIR / "scrcpy.exe",
        *sorted(BASE_DIR.glob("scrcpy-*/scrcpy.exe")),  # scrcpy-win64-vX.X.X/
    ]
    found = next((p for p in candidates if p.exists()), None)
    scrcpy_cmd = str(found) if found else "scrcpy"
    try:
        cmd = [scrcpy_cmd, "--no-audio", "--max-fps", "30", "--always-on-top",
               "--window-title", "scrcpy-drone"]
        if device_serial:
            cmd += ["-s", device_serial]
        _scrcpy_proc = _sp.Popen(cmd, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        return JSONResponse(content={"ok": True})
    except FileNotFoundError:
        return JSONResponse(content={
            "ok": False,
            "error": "scrcpy not found. Download scrcpy.exe and place it in the Demo/ folder."
        })
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)})


@app.get("/detect_scrcpy")
async def detect_scrcpy():
    """Auto-detect scrcpy window position and size."""
    try:
        windows = _find_windows("scrcpy")
        if not windows:
            return JSONResponse(content={"found": False, "windows": [],
                                         "error": "No scrcpy window found. Is scrcpy running?"})
        # Return all matches, frontend picks first or lets user choose
        safe = [{"title": w["title"], "hwnd": w["hwnd"],
                 "x": w["x"], "y": w["y"],
                 "w": w["w"], "h": w["h"]} for w in windows]
        return JSONResponse(content={"found": True, "windows": safe})
    except Exception as e:
        return JSONResponse(content={"found": False, "windows": [],
                                     "error": str(e)})


def _run_adb(*args, timeout=10):
    """Run adb command, return (stdout, stderr, ok)."""
    import subprocess
    cmd = ["adb"] + list(args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.stderr.strip(), r.returncode == 0
    except FileNotFoundError:
        return "", "adb not found in PATH", False
    except subprocess.TimeoutExpired:
        return "", "timeout", False


@app.post("/adb_pair")
async def adb_pair(request: Request):
    """Pair with phone via ADB wireless.
    adb pair <ip>:<port> prompts for code via stdin -- dung Popen + communicate.
    """
    import subprocess as _sp
    try:
        body = await request.json()
        ip = body.get("ip", "").strip()
        port = str(body.get("port", "")).strip()
        code = str(body.get("code", "")).strip()
        if not ip or not port or not code:
            return JSONResponse(content={"ok": False, "output": "ip, port, code required"})
        addr = f"{ip}:{port}"
        proc = _sp.Popen(
            ["adb", "pair", addr],
            stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.PIPE,
        )
        stdout_b, stderr_b = proc.communicate(input=(code + "\n").encode(), timeout=15)
        output = (stdout_b.decode(errors="replace") + " " + stderr_b.decode(errors="replace")).strip()
        success = "successfully" in output.lower() or "paired" in output.lower()
        return JSONResponse(content={"ok": success, "output": output or "(no output)"})
    except FileNotFoundError:
        return JSONResponse(content={"ok": False, "output": "adb not found in PATH"})
    except _sp.TimeoutExpired:
        try: proc.kill()
        except Exception: pass
        return JSONResponse(content={"ok": False, "output": "timeout — check pair port"})
    except Exception as exc:
        return JSONResponse(content={"ok": False, "output": f"error: {exc}"})


@app.post("/adb_connect")
async def adb_connect(request: Request):
    """Connect to phone via ADB wireless: adb connect <ip>:<port>."""
    body = await request.json()
    ip = body.get("ip", "").strip()
    port = str(body.get("port", "")).strip()
    if not ip or not port:
        return JSONResponse(content={"ok": False, "output": "ip, port required"})
    addr = f"{ip}:{port}"
    stdout, stderr, _ = _run_adb("connect", addr)
    output = (stdout + " " + stderr).strip()
    success = "connected" in output.lower() and "unable" not in output.lower()
    return JSONResponse(content={"ok": success, "output": output or "(no output)"})


@app.post("/adb_preview_start")
async def adb_preview_start(request: Request):
    """Bat dau preview-only worker: hien thi man hinh dien thoai ma khong inference."""
    global adb_preview_mode
    body = await request.json()
    device_serial = body.get("device_serial", "").strip()
    if not device_serial:
        return JSONResponse(content={"ok": False, "error": "device_serial required"})
    adb_preview_mode = False  # Dung worker cu neu co
    time.sleep(0.2)
    adb_preview_mode = True
    t = threading.Thread(
        target=adb_preview_only_worker, args=(device_serial,),
        daemon=True, name="adb_preview",
    )
    t.start()
    return JSONResponse(content={"ok": True})


@app.post("/adb_preview_stop")
async def adb_preview_stop():
    """Dung preview-only worker (ca ADB va scrcpy)."""
    global adb_preview_mode
    adb_preview_mode = False
    return JSONResponse(content={"ok": True})


@app.post("/scrcpy_preview_start")
async def scrcpy_preview_start(request: Request):
    """Bat dau preview scrcpy window vao main video panel (qua adb_preview_queue)."""
    global adb_preview_mode
    body = await request.json()
    hwnd = body.get("hwnd")
    if not hwnd:
        return JSONResponse(content={"ok": False, "error": "hwnd required"})
    adb_preview_mode = False
    time.sleep(0.15)
    adb_preview_mode = True
    t = threading.Thread(
        target=scrcpy_preview_only_worker, args=(int(hwnd),),
        daemon=True, name="scrcpy_preview",
    )
    t.start()
    return JSONResponse(content={"ok": True})


@app.get("/adb_screen_feed")
async def adb_screen_feed():
    """MJPEG stream: raw phone screen preview (khong annotate)."""
    return StreamingResponse(
        generate_adb_preview(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/adb_status")
async def adb_status():
    """Check ADB device list. Tra ve serial (ip:port) de dung cho screen capture."""
    stdout, stderr, ok = _run_adb("devices")
    lines = [l.strip() for l in stdout.splitlines() if l.strip() and "List of" not in l]
    devices = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].strip() == "device":
            devices.append({"serial": parts[0], "state": parts[1]})
    return JSONResponse(content={
        "devices": devices,
        "count": len(devices),
        "ok": ok,
    })


@app.post("/start_live")
async def start_live(request: Request):
    """Start live mode: screen capture + ADB telemetry."""
    global live_mode, live_thread, telemetry_poller, is_running, current_session

    body = await request.json()
    adb_device = body.get("adb_device")  # "ip:port" — ADB screen capture (preferred)
    hwnd = body.get("hwnd")              # Win32 HWND — scrcpy background capture

    # Stop video playback if running
    is_running = False
    live_mode = False
    time.sleep(0.3)

    # Clear queues
    for q in (capture_queue, adb_preview_queue):
        while not q.empty():
            try:
                q.get_nowait()
            except Empty:
                break

    # New session
    current_session = f"live_{int(time.time())}"
    with state_lock:
        flagged_detections.clear()
        detection_history.clear()
        detection_timeline.clear()
        detection_saves.clear()
    with frame_lock:
        latest_frame = None

    # Start telemetry poller
    if telemetry_poller:
        telemetry_poller.stop()
    try:
        from telemetry import TelemetryPoller
        telemetry_poller = TelemetryPoller(
            dji_log_exe=DJI_LOG_EXE,
            api_key=DJI_API_KEY,
            adb_device=adb_device or None,
            poll_interval=3.0,
        )
        telemetry_poller.start()
    except Exception as e:
        print(f"Telemetry start failed: {e}")
        telemetry_poller = None

    # Priority: adb_device > hwnd > mss
    live_mode = True
    if adb_device:
        crop = body.get("crop")  # {x, y, w, h} optional
        live_thread = threading.Thread(
            target=adb_screen_capture_worker,
            args=(adb_device, crop),
            daemon=True, name="live_capture",
        )
        capture_info = {"mode": "adb", "device": adb_device, "crop": crop}
    elif hwnd:
        w = int(body.get("w", 1280))
        h = int(body.get("h", 720))
        live_thread = threading.Thread(
            target=live_capture_worker_hwnd,
            args=(int(hwnd), w, h),
            daemon=True, name="live_capture",
        )
        capture_info = {"mode": "hwnd", "hwnd": hwnd, "w": w, "h": h}
    else:
        monitor_idx = int(body.get("monitor", 1))
        crop = body.get("crop", {})
        crop_region = {
            "left": int(crop.get("x", 0)),
            "top": int(crop.get("y", 0)),
            "width": int(crop.get("w", 1280)),
            "height": int(crop.get("h", 720)),
        }
        live_thread = threading.Thread(
            target=live_capture_worker,
            args=(monitor_idx, crop_region),
            daemon=True, name="live_capture",
        )
        capture_info = {"mode": "mss", "monitor": monitor_idx, "crop": crop_region}
    live_thread.start()

    return {
        "status": "ok",
        "mode": "live",
        "capture": capture_info,
        "telemetry": telemetry_poller is not None,
        "session": current_session,
    }


@app.post("/stop_live")
async def stop_live():
    """Stop live mode, return to upload mode."""
    global live_mode, telemetry_poller

    live_mode = False
    if telemetry_poller:
        telemetry_poller.stop()
        telemetry_poller = None

    # Clear queue
    while not capture_queue.empty():
        try:
            capture_queue.get_nowait()
        except Empty:
            break

    with frame_lock:
        latest_frame = None

    return {"status": "ok", "mode": "upload"}


@app.get("/telemetry")
async def get_telemetry():
    """Return current telemetry data from ADB flight record."""
    if telemetry_poller is None:
        return JSONResponse(content={
            "connected": False,
            "live_mode": live_mode,
            "data": None,
            "error": "Telemetry not started",
        })
    return JSONResponse(content={
        "connected": telemetry_poller.is_connected,
        "live_mode": live_mode,
        "data": telemetry_poller.get_latest(),
        "error": telemetry_poller.last_error,
    })


def generate_frames():
    """MJPEG stream -- lay latest_frame tu inference thread, khong xu ly gi nang."""
    global video_fps

    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            # Placeholder
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Upload a video to start", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            _, buf = cv2.imencode('.jpg', blank)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.5)
            continue

        t0 = time.time()

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

        elapsed = time.time() - t0
        with state_lock:
            video_fps = 1.0 / max(elapsed, 1e-6)

        # ~30 FPS max cho MJPEG stream
        time.sleep(max(0, 1.0 / 30.0 - elapsed))


# =====================================================================
# STARTUP
# =====================================================================
@app.on_event("startup")
async def startup():
    load_model()

    # Start capture + inference threads
    t1 = threading.Thread(target=capture_worker, daemon=True, name="capture")
    t2 = threading.Thread(target=inference_worker, daemon=True, name="inference")
    t1.start()
    t2.start()
    print("Capture + Inference threads started.")
