"""
Export Mid-fusion Progressive model sang ONNX.
Chay: cd Demo && python export_onnx.py

Output: models/fusion_progressive.onnx
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Conv, Detect

# =====================================================================
# CONFIG
# =====================================================================
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(BASE_DIR)

CHECKPOINT = os.path.join(ROOT_DIR, 'Mid-fusion', 'outputs', 'progressive_luong1', 'seed_42', 'fusion_best.pt')
RGB_BB_PATH = os.path.join(ROOT_DIR, 'backbones', 'llvip_rgb_best.pt')
THR_BB_PATH = os.path.join(ROOT_DIR, 'backbones', 'llvip_thermal_best.pt')
OUTPUT_ONNX = os.path.join(BASE_DIR, 'models', 'fusion_progressive.onnx')
IMG_SIZE = 640

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================================================
# MODEL -- dung dung class tu ultralytics nhu notebook training
# =====================================================================
class RGBTFusionDetectorProgressive(nn.Module):
    """Copy tu notebook Mid_Progressive_Luong1.ipynb, chi sua forward() de concat output cho ONNX."""

    EXTRACT_LAYERS = {4: 64, 6: 128, 9: 256}

    def __init__(self, rgb_backbone, thermal_backbone, nc=1, freeze_backbones=True):
        super().__init__()
        self.rgb_stream = rgb_backbone
        self.thermal_stream = thermal_backbone

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

    def _extract_features(self, stream, x):
        feats = {}
        for i, layer in enumerate(stream):
            x = layer(x)
            if i in self.EXTRACT_LAYERS:
                feats[i] = x
        return feats

    def forward(self, rgb, thermal):
        rgb_f = self._extract_features(self.rgb_stream, rgb)
        thr_f = self._extract_features(self.thermal_stream, thermal)

        p3 = self.fuse_p3(torch.cat([rgb_f[4], thr_f[4]], dim=1))
        p4 = self.fuse_p4(torch.cat([rgb_f[6], thr_f[6]], dim=1))
        p5 = self.fuse_p5(torch.cat([rgb_f[9], thr_f[9]], dim=1))

        p4_td = self.td_c2f_p4(torch.cat([self.upsample(p5), p4], dim=1))
        p3_out = self.td_c2f_p3(torch.cat([self.upsample(p4_td), p3], dim=1))

        p4_out = self.bu_c2f_p4(torch.cat([self.bu_conv_p4(p3_out), p4_td], dim=1))
        p5_out = self.bu_c2f_p5(torch.cat([self.bu_conv_p5(p4_out), p5], dim=1))

        # Goi detect nhung bypass postprocess -- chi lay raw output
        outs = self.detect([p3_out, p4_out, p5_out])
        # Neu Detect tra ve tuple/list of tensors [B, no, Hi, Wi]
        if isinstance(outs, (list, tuple)):
            flat = [o.flatten(2) for o in outs]
            return torch.cat(flat, dim=2)  # [B, no, total_anchors]
        # Neu Detect da concat san
        return outs


# =====================================================================
# EXPORT
# =====================================================================
if __name__ == '__main__':
    print(f'Device: {device}')
    print(f'Loading backbones...')

    rgb_bb = nn.ModuleList(list(YOLO(RGB_BB_PATH).model.model)[:10]).to(device)
    thr_bb = nn.ModuleList(list(YOLO(THR_BB_PATH).model.model)[:10]).to(device)

    print(f'Building model...')
    model = RGBTFusionDetectorProgressive(rgb_bb, thr_bb, nc=1, freeze_backbones=True)
    model = model.to(device)

    print(f'Loading checkpoint: {CHECKPOINT}')
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Tat Detect postprocess (export mode)
    model.detect.export = True
    model.detect.format = 'onnx'

    # Dummy inputs
    dummy_rgb = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    dummy_thr = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    print(f'Exporting to ONNX: {OUTPUT_ONNX}')
    os.makedirs(os.path.dirname(OUTPUT_ONNX), exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_rgb, dummy_thr),
        OUTPUT_ONNX,
        input_names=['rgb', 'thermal'],
        output_names=['output'],
        dynamic_axes={
            'rgb': {0: 'batch'},
            'thermal': {0: 'batch'},
            'output': {0: 'batch'},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify
    import onnx
    onnx_model = onnx.load(OUTPUT_ONNX)
    onnx.checker.check_model(onnx_model)

    file_size = os.path.getsize(OUTPUT_ONNX) / (1024 * 1024)
    print(f'Export OK: {OUTPUT_ONNX} ({file_size:.1f} MB)')
    print(f'Input: rgb [1, 3, {IMG_SIZE}, {IMG_SIZE}], thermal [1, 3, {IMG_SIZE}, {IMG_SIZE}]')
    print(f'Output: [1, no, total_anchors]')
