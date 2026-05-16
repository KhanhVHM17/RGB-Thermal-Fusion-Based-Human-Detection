"""
Fusion Detector + GPS estimation cho Demo web app.
Tai dung logic tu GPS_Test.ipynb va Mid-fusion notebooks.

Supports:
  - ONNX Runtime (onnx mode)       -- lightweight, chay local CPU/GPU
  - Mid-fusion (mid mode)           -- torch, can GPU manh
  - Early/Late fusion (yolo mode)   -- backup plan
"""

import math
import copy
import threading
import cv2
import numpy as np
from pathlib import Path

# Torch is optional — only needed for mode='mid'
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Optional reverse geocoding
try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    _geolocator = Nominatim(user_agent="drone_demo_app", timeout=3)
    _reverse_geo = RateLimiter(_geolocator.reverse, min_delay_seconds=1)
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False

# =====================================================================
# GPS MATH (tu GPS_Test.ipynb)
# =====================================================================
def deg2rad(x):
    return x * math.pi / 180.0

def compute_intrinsics_from_fov(w, h, hfov_deg, vfov_deg):
    fx = (w / 2.0) / math.tan(deg2rad(hfov_deg) / 2.0)
    fy = (h / 2.0) / math.tan(deg2rad(vfov_deg) / 2.0)
    return fx, fy, w / 2.0, h / 2.0

def pixel_to_camera_ray(u, v, fx, fy, cx, cy):
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)

def rotation_matrix_ypr(yaw_deg, pitch_deg, roll_deg):
    yaw, pitch, roll = deg2rad(yaw_deg), deg2rad(pitch_deg), deg2rad(roll_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return Rz @ Ry @ Rx

def intersect_ray_ground(ray_world, cam_xyz):
    if abs(ray_world[2]) < 1e-8:
        return None
    t = -cam_xyz[2] / ray_world[2]
    return cam_xyz + t * ray_world if t > 0 else None

def meters_to_latlon(dx_east, dy_north, lat0, lon0):
    dlat = dy_north / 111320.0
    dlon = dx_east / (111320.0 * math.cos(math.radians(lat0)))
    return lat0 + dlat, lon0 + dlon

def estimate_target_gps(x1, y1, x2, y2, img_w, img_h, drone_config):
    u = (x1 + x2) / 2.0
    v = float(y2)  # bottom center
    fx, fy, cx, cy = compute_intrinsics_from_fov(
        img_w, img_h, drone_config['hfov'], drone_config['vfov'])
    ray_cam = pixel_to_camera_ray(u, v, fx, fy, cx, cy)
    cam_world = np.array([0.0, 0.0, drone_config['alt']], dtype=np.float64)
    R = rotation_matrix_ypr(drone_config['yaw'], drone_config['pitch'], drone_config.get('roll', 0))
    ray_world = R @ ray_cam
    gp = intersect_ray_ground(ray_world, cam_world)
    if gp is None:
        return None
    dx, dy = float(gp[0]), float(gp[1])
    lat, lon = meters_to_latlon(dx, dy, drone_config['lat'], drone_config['lon'])
    return {'lat': lat, 'lon': lon, 'dx_east': dx, 'dy_north': dy}


def compute_gsd(alt, hfov_deg, vfov_deg, img_w, img_h):
    """Ground Sampling Distance (cm/pixel).
    Tinh GSD theo huong ngang va doc, tra ve max (worst-case).
    Dua tren paper Dumencic et al. — recall giam manh khi GSD > 4 cm/px.
    """
    gsd_h = (2.0 * alt * math.tan(deg2rad(hfov_deg) / 2.0)) / img_w * 100.0
    gsd_v = (2.0 * alt * math.tan(deg2rad(vfov_deg) / 2.0)) / img_h * 100.0
    return round(max(gsd_h, gsd_v), 2)


_geocode_cache = {}
_geocode_lock = threading.Lock()
_geocode_pending = {}  # key -> True (dang query)


def reverse_geocode(lat, lon):
    """Non-blocking geocode: tra ve cache ngay, neu chua co thi fire background thread."""
    if not HAS_GEOPY:
        return ""
    key = (round(lat, 4), round(lon, 4))
    with _geocode_lock:
        if key in _geocode_cache:
            return _geocode_cache[key]
        if key in _geocode_pending:
            return "resolving..."
        _geocode_pending[key] = True

    # Fire background thread de query, khong block detection loop
    def _do_geocode():
        try:
            loc = _reverse_geo((lat, lon), language="en", exactly_one=True)
            if loc is None:
                result = "unknown area"
            else:
                raw = loc.raw.get("address", {})
                parts = [raw.get(k, "") for k in
                         ["road", "neighbourhood", "suburb", "city_district", "city", "state", "country"]]
                result = ", ".join(p for p in parts if p) or loc.address
        except Exception:
            result = "geocode failed"
        with _geocode_lock:
            _geocode_cache[key] = result
            _geocode_pending.pop(key, None)

    t = threading.Thread(target=_do_geocode, daemon=True)
    t.start()
    return "resolving..."

def google_maps_link(lat, lon):
    return f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"


# =====================================================================
# NUMPY NMS (cho ONNX, khong can torch)
# =====================================================================
def _nms_numpy(boxes, scores, iou_thres):
    """Pure numpy NMS."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return np.array(keep)


def postprocess_onnx(output, conf_thres=0.25, iou_thres=0.45, nc=1, img_size=640):
    """Decode ONNX output [1, 4+nc, total_anchors] -> array of [x1,y1,x2,y2,conf,cls].

    Detect head with export=True outputs decoded boxes (xywh) + class scores.
    Shape: [1, 4+nc, 8400] where 4 = xywh, nc = class scores (already sigmoided).
    """
    pred = output[0]  # [4+nc, N]
    boxes_xywh = pred[:4, :].T  # [N, 4] -- x_center, y_center, w, h
    cls_scores = pred[4:4+nc, :].T  # [N, nc]

    conf = cls_scores.max(axis=1)  # [N]
    cls_id = cls_scores.argmax(axis=1)  # [N]

    mask = conf > conf_thres
    if not mask.any():
        return None

    bxywh = boxes_xywh[mask]  # [M, 4]
    conf_f = conf[mask]
    cls_f = cls_id[mask]

    # xywh -> xyxy
    x1 = bxywh[:, 0] - bxywh[:, 2] / 2
    y1 = bxywh[:, 1] - bxywh[:, 3] / 2
    x2 = bxywh[:, 0] + bxywh[:, 2] / 2
    y2 = bxywh[:, 1] + bxywh[:, 3] / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = _nms_numpy(boxes, conf_f, iou_thres)
    return np.concatenate([boxes[keep], conf_f[keep, None], cls_f[keep, None].astype(np.float32)], axis=1)


# =====================================================================
# CUSTOM NMS (tu Mid-fusion notebooks, can torch)
# =====================================================================
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def custom_nms(prediction, conf_thres=0.25, iou_thres=0.45, nc=1):
    bs = prediction.shape[0]
    output = [None] * bs
    for bi in range(bs):
        pred = prediction[bi]
        if pred.shape[0] == 0:
            continue
        reg = pred[:, :64]
        cls_scores = pred[:, 64:64+nc].sigmoid()
        conf, cls_id = cls_scores.max(dim=1)
        mask = conf > conf_thres
        if not mask.any():
            continue
        pred_f = pred[mask]
        conf_f = conf[mask]
        cls_f = cls_id[mask].float()
        # DFL decode
        reg_f = pred_f[:, :64].view(-1, 4, 16).softmax(dim=2)
        arange = torch.arange(16, device=pred.device, dtype=torch.float32)
        reg_f = (reg_f * arange).sum(dim=2)
        # anchors
        H = W = 0
        strides = [8, 16, 32]
        img_size = 640
        anchors_list = []
        strides_list = []
        for s in strides:
            gs = img_size // s
            yy, xx = torch.meshgrid(torch.arange(gs, device=pred.device),
                                     torch.arange(gs, device=pred.device), indexing='ij')
            anchors_list.append(torch.stack([xx.flatten(), yy.flatten()], dim=1).float() + 0.5)
            strides_list.append(torch.full((gs * gs,), s, device=pred.device, dtype=torch.float32))
        all_anchors = torch.cat(anchors_list, dim=0)
        all_strides = torch.cat(strides_list, dim=0)
        anc = all_anchors[mask]
        st = all_strides[mask]
        x1 = (anc[:, 0] - reg_f[:, 0]) * st
        y1 = (anc[:, 1] - reg_f[:, 1]) * st
        x2 = (anc[:, 0] + reg_f[:, 2]) * st
        y2 = (anc[:, 1] + reg_f[:, 3]) * st
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        keep = torch.ops.torchvision.nms(boxes, conf_f, iou_thres)
        det = torch.cat([boxes[keep], conf_f[keep].unsqueeze(1), cls_f[keep].unsqueeze(1)], dim=1)
        output[bi] = det
    return output


# =====================================================================
# MID-FUSION MODEL DEFINITION (tu notebooks)
# =====================================================================
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2 if p is None else p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c2, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1)
        self.m = nn.ModuleList(
            nn.Sequential(Conv(c_, c_, 3), Conv(c_, c_, 3)) for _ in range(n))
        self.shortcut = shortcut
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Detect(nn.Module):
    def __init__(self, nc=1, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3), Conv(x, x, 3)),
                nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3)),
                nn.Conv2d(c3, nc, 1)) for x in ch)
        self.dfl = nn.Conv2d(self.reg_max, 1, 1, bias=False)
        self.dfl.weight.data[:] = (
            torch.arange(self.reg_max, dtype=torch.float).view(1, -1, 1, 1) / self.reg_max)
    def forward(self, x):
        return [torch.cat([self.cv2[i](x[i]), self.cv3[i](x[i])], 1) for i in range(self.nl)]

class RGBTFusionDetector(nn.Module):
    """Baseline Mid-fusion: Concat + Conv1x1, ch=(64,128,256)"""
    def __init__(self, rgb_bb, thr_bb, nc=1, freeze_backbones=True):
        super().__init__()
        self.rgb_stream = copy.deepcopy(rgb_bb)
        self.thermal_stream = copy.deepcopy(thr_bb)
        if freeze_backbones:
            for p in self.rgb_stream.parameters():
                p.requires_grad = False
            for p in self.thermal_stream.parameters():
                p.requires_grad = False
        self.fuse_p3 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.SiLU())
        self.fuse_p4 = nn.Sequential(nn.Conv2d(256, 128, 1, bias=False), nn.BatchNorm2d(128), nn.SiLU())
        self.fuse_p5 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False), nn.BatchNorm2d(256), nn.SiLU())
        self.td_c2f_p4 = C2f(128 + 256, 128, n=1)
        self.td_c2f_p3 = C2f(64 + 128, 64, n=1)
        self.bu_conv_p4 = Conv(64, 128, 3, 2)
        self.bu_c2f_p4 = C2f(128 + 128, 128, n=1)
        self.bu_conv_p5 = Conv(128, 256, 3, 2)
        self.bu_c2f_p5 = C2f(256 + 256, 256, n=1)
        self.detect = Detect(nc=nc, ch=(64, 128, 256))
        self.detect.stride = torch.tensor([8., 16., 32.])

    def _extract(self, bb, x):
        feats = {}
        for i, layer in enumerate(bb):
            x = layer(x)
            if i in [4, 6, 9]:
                feats[i] = x
        return feats

    def forward(self, rgb, thr):
        rf = self._extract(self.rgb_stream, rgb)
        tf = self._extract(self.thermal_stream, thr)
        p3 = self.fuse_p3(torch.cat([rf[4], tf[4]], 1))
        p4 = self.fuse_p4(torch.cat([rf[6], tf[6]], 1))
        p5 = self.fuse_p5(torch.cat([rf[9], tf[9]], 1))
        p4 = self.td_c2f_p4(torch.cat([F.interpolate(p5, scale_factor=2), p4], 1))
        p3 = self.td_c2f_p3(torch.cat([F.interpolate(p4, scale_factor=2), p3], 1))
        p4 = self.bu_c2f_p4(torch.cat([self.bu_conv_p4(p3), p4], 1))
        p5 = self.bu_c2f_p5(torch.cat([self.bu_conv_p5(p4), p5], 1))
        return self.detect([p3, p4, p5])

class RGBTFusionDetectorProgressive(nn.Module):
    """Progressive Mid-fusion: Concat + Conv1x1, ch=(128,256,512)"""
    EXTRACT_LAYERS = {4: 64, 6: 128, 9: 256}

    def __init__(self, rgb_bb, thr_bb, nc=1, freeze_backbones=True):
        super().__init__()
        self.rgb_stream = copy.deepcopy(rgb_bb)
        self.thermal_stream = copy.deepcopy(thr_bb)
        if freeze_backbones:
            for p in self.rgb_stream.parameters():
                p.requires_grad = False
            for p in self.thermal_stream.parameters():
                p.requires_grad = False
        self.fuse_p3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.SiLU())
        self.fuse_p4 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.SiLU())
        self.fuse_p5 = nn.Sequential(nn.Conv2d(512, 512, 1, bias=False), nn.BatchNorm2d(512), nn.SiLU())
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.td_c2f_p4 = C2f(512 + 256, 256, n=1, shortcut=False)
        self.td_c2f_p3 = C2f(256 + 128, 128, n=1, shortcut=False)
        self.bu_conv_p4 = Conv(128, 128, 3, 2)
        self.bu_c2f_p4 = C2f(128 + 256, 256, n=1, shortcut=False)
        self.bu_conv_p5 = Conv(256, 256, 3, 2)
        self.bu_c2f_p5 = C2f(256 + 512, 512, n=1, shortcut=False)
        self.detect = Detect(nc=nc, ch=(128, 256, 512))
        self.detect.stride = torch.tensor([8., 16., 32.])

    def _extract(self, stream, x):
        feats = {}
        for i, layer in enumerate(stream):
            x = layer(x)
            if i in self.EXTRACT_LAYERS:
                feats[i] = x
        return feats

    def forward(self, rgb, thermal):
        rf = self._extract(self.rgb_stream, rgb)
        tf = self._extract(self.thermal_stream, thermal)
        p3 = self.fuse_p3(torch.cat([rf[4], tf[4]], dim=1))
        p4 = self.fuse_p4(torch.cat([rf[6], tf[6]], dim=1))
        p5 = self.fuse_p5(torch.cat([rf[9], tf[9]], dim=1))
        p4_td = self.td_c2f_p4(torch.cat([self.upsample(p5), p4], dim=1))
        p3_out = self.td_c2f_p3(torch.cat([self.upsample(p4_td), p3], dim=1))
        p4_out = self.bu_c2f_p4(torch.cat([self.bu_conv_p4(p3_out), p4_td], dim=1))
        p5_out = self.bu_c2f_p5(torch.cat([self.bu_conv_p5(p4_out), p5], dim=1))
        return self.detect([p3_out, p4_out, p5_out])


ARCH_MAP = {
    'baseline': RGBTFusionDetector,
    'progressive': RGBTFusionDetectorProgressive,
}


# =====================================================================
# DETECTOR CLASS
# =====================================================================
class FusionDetector:
    """Wraps Mid-fusion or YOLO model for demo inference."""

    def __init__(self, model_path, rgb_bb_path=None, thr_bb_path=None,
                 mode='onnx', arch='progressive', device=None, img_size=640):
        """
        Args:
            model_path: path to .onnx (onnx), fusion_best.pt (mid), or best.pt (yolo)
            rgb_bb_path: path to RGB backbone .pt (mid-fusion only)
            thr_bb_path: path to Thermal backbone .pt (mid-fusion only)
            mode: 'onnx' (recommended), 'mid' (torch), 'yolo' (YOLO API)
            arch: 'baseline' or 'progressive' (mid-fusion architecture)
            device: torch device (mid/yolo only)
            img_size: input size
        """
        self.mode = mode
        self.arch = arch
        self.img_size = img_size

        # Drone GPS config — starts empty, populated when processing begins
        self.drone_config = {
            'lat': 0, 'lon': 0,
            'alt': 0, 'yaw': 0, 'pitch': -90.0, 'roll': 0.0,
            'hfov': 69.0, 'vfov': 42.0,
        }

        if mode == 'onnx':
            self.device = 'onnx'
            self._load_onnx(model_path)
        elif mode == 'mid':
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._load_mid_fusion(model_path, rgb_bb_path, thr_bb_path)
        else:
            self.device = device or 'cpu'
            self._load_yolo(model_path)

        print(f"Model loaded ({mode}) on {self.device}")

    def _load_onnx(self, model_path):
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(model_path, providers=providers)
        used = self.ort_session.get_providers()
        print(f"ONNX providers: {used}")

    def _load_mid_fusion(self, model_path, rgb_bb_path, thr_bb_path):
        from ultralytics import YOLO
        ModelClass = ARCH_MAP.get(self.arch, RGBTFusionDetectorProgressive)
        rgb_bb = nn.ModuleList(list(YOLO(rgb_bb_path).model.model)[:10]).to(self.device)
        thr_bb = nn.ModuleList(list(YOLO(thr_bb_path).model.model)[:10]).to(self.device)
        self.model = ModelClass(rgb_bb, thr_bb, nc=1, freeze_backbones=True)
        self.model = self.model.to(self.device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        state = ckpt['model_state_dict']
        self.model.load_state_dict(state)
        self.model.eval()
        del rgb_bb, thr_bb

    def _load_yolo(self, model_path):
        from ultralytics import YOLO
        self.yolo_model = YOLO(model_path)

    def update_drone_config(self, **kwargs):
        self.drone_config.update(kwargs)

    def _preprocess_np(self, frame):
        """Preprocess RGB frame -> [1, 3, H, W] numpy float32."""
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (img.transpose(2, 0, 1).astype(np.float32) / 255.0)[np.newaxis]

    def _preprocess_thermal_np(self, frame):
        """Preprocess thermal frame -> [1, 3, H, W] numpy float32."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        gray = cv2.resize(gray, (self.img_size, self.img_size))
        ch1 = (gray.astype(np.float32) / 255.0)[np.newaxis]  # [1, H, W]
        return np.stack([ch1[0]] * 3, axis=0)[np.newaxis]     # [1, 3, H, W]

    def _preprocess(self, frame):
        if self.mode == 'onnx':
            return self._preprocess_np(frame)
        t = torch.from_numpy(self._preprocess_np(frame))
        return t.to(self.device)

    def _preprocess_thermal(self, frame):
        if self.mode == 'onnx':
            return self._preprocess_thermal_np(frame)
        t = torch.from_numpy(self._preprocess_thermal_np(frame))
        return t.to(self.device)

    def detect_frame(self, rgb_frame, thr_frame=None, conf=0.25, iou=0.45):
        """
        Run detection on a frame pair.
        Args:
            rgb_frame: BGR numpy array
            thr_frame: BGR/grayscale numpy array (bat buoc cho fusion mode)
        Returns:
            list of dicts with detection info + GPS
        """
        img_h, img_w = rgb_frame.shape[:2]

        if self.mode == 'yolo':
            return self._detect_yolo(rgb_frame, img_w, img_h, conf)

        # Fusion modes (onnx/mid) — thermal bat buoc
        if thr_frame is None:
            raise ValueError("Thermal frame is required for RGBT fusion mode")

        if self.mode == 'onnx':
            return self._detect_onnx(rgb_frame, thr_frame, img_w, img_h, conf, iou)
        else:
            return self._detect_mid(rgb_frame, thr_frame, img_w, img_h, conf, iou)

    def _detect_onnx(self, rgb_frame, thr_frame, img_w, img_h, conf, iou):
        rgb_np = self._preprocess_np(rgb_frame)
        thr_np = self._preprocess_thermal_np(thr_frame)
        output = self.ort_session.run(None, {'rgb': rgb_np, 'thermal': thr_np})
        dets = postprocess_onnx(output[0], conf_thres=conf, iou_thres=iou,
                                nc=1, img_size=self.img_size)
        if dets is None or len(dets) == 0:
            return []

        results = []
        scale_x = img_w / self.img_size
        scale_y = img_h / self.img_size
        for idx, det in enumerate(dets, start=1):
            x1, y1, x2, y2, sc, cl = det
            ox1, oy1 = int(x1 * scale_x), int(y1 * scale_y)
            ox2, oy2 = int(x2 * scale_x), int(y2 * scale_y)
            gps = estimate_target_gps(ox1, oy1, ox2, oy2, img_w, img_h, self.drone_config)
            entry = {
                'id': idx, 'conf': float(sc),
                'bbox': [ox1, oy1, ox2, oy2],
                'lat': None, 'lon': None,
                'address': '', 'gmaps': '',
            }
            if gps:
                entry['lat'] = gps['lat']
                entry['lon'] = gps['lon']
                entry['address'] = reverse_geocode(gps['lat'], gps['lon'])
                entry['gmaps'] = google_maps_link(gps['lat'], gps['lon'])
            results.append(entry)
        return results

    def _detect_mid(self, rgb_frame, thr_frame, img_w, img_h, conf, iou):
        rgb_t = self._preprocess(rgb_frame)
        thr_t = self._preprocess_thermal(thr_frame)

        with torch.no_grad():
            preds_raw = self.model(rgb_t, thr_t)
            pred = preds_raw[0] if isinstance(preds_raw, (list, tuple)) else preds_raw
            dets = custom_nms(pred, conf_thres=conf, iou_thres=iou, nc=1)[0]

        if dets is None or len(dets) == 0:
            return []

        results = []
        scale_x = img_w / self.img_size
        scale_y = img_h / self.img_size

        for idx, det in enumerate(dets.cpu().numpy(), start=1):
            x1, y1, x2, y2, sc, cl = det
            # Scale back to original image coords
            ox1, oy1 = int(x1 * scale_x), int(y1 * scale_y)
            ox2, oy2 = int(x2 * scale_x), int(y2 * scale_y)

            gps = estimate_target_gps(ox1, oy1, ox2, oy2, img_w, img_h, self.drone_config)
            entry = {
                'id': idx, 'conf': float(sc),
                'bbox': [ox1, oy1, ox2, oy2],
                'lat': None, 'lon': None,
                'address': '', 'gmaps': '',
            }
            if gps:
                entry['lat'] = gps['lat']
                entry['lon'] = gps['lon']
                entry['address'] = reverse_geocode(gps['lat'], gps['lon'])
                entry['gmaps'] = google_maps_link(gps['lat'], gps['lon'])
            results.append(entry)
        return results

    def _detect_yolo(self, rgb_frame, img_w, img_h, conf):
        results = self.yolo_model.predict(source=rgb_frame, conf=conf, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        detections = []
        for i, (box, c, cls) in enumerate(
                zip(r.boxes.xyxy.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.cls.cpu().numpy()), start=1):
            x1, y1, x2, y2 = box.astype(int)
            gps = estimate_target_gps(x1, y1, x2, y2, img_w, img_h, self.drone_config)
            entry = {
                'id': i, 'conf': float(c),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'lat': None, 'lon': None,
                'address': '', 'gmaps': '',
            }
            if gps:
                entry['lat'] = gps['lat']
                entry['lon'] = gps['lon']
                entry['address'] = reverse_geocode(gps['lat'], gps['lon'])
                entry['gmaps'] = google_maps_link(gps['lat'], gps['lon'])
            detections.append(entry)
        return detections

    def draw_detections(self, frame, detections):
        """Draw boxes + labels on frame. Returns annotated copy."""
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            sc = d['conf']
            idx = d['id']

            # Box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Bottom center dot
            cx, cy = (x1 + x2) // 2, y2
            cv2.circle(out, (cx, cy), 4, (255, 0, 255), -1)

            # Label: #N (conf)
            label = f"#{idx} ({sc:.2f})"
            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ly = max(lh + 4, y1 - 4)
            cv2.rectangle(out, (x1, ly - lh - 4), (x1 + lw + 6, ly + bl), (0, 200, 0), -1)
            cv2.putText(out, label, (x1 + 2, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # GPS label below box
            if d['lat'] is not None:
                gps_label = f"#{idx}: {d['lat']:.5f}, {d['lon']:.5f}"
                y_text = min(frame.shape[0] - 10, y2 + 20)
                cv2.putText(out, gps_label, (x1, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
        return out
