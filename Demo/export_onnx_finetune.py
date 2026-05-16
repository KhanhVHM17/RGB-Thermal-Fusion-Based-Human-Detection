"""
Export finetuned Progressive S2 model sang ONNX.

Chay sau khi da scp ft_best.pt ve local:
  scp root@<server>:/root/AIP491/Finetune/outputs/progressive_s2_custom_ntut/seed_<best>/ft_best.pt .

Chay:
  cd Demo
  python export_onnx_finetune.py

Output: Demo/models/fusion_progressive_finetune.onnx
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Conv, Detect

# =====================================================================
# CONFIG -- chinh CHECKPOINT sau khi scp ve
# =====================================================================
BASE_DIR  = os.path.dirname(__file__)
ROOT_DIR  = os.path.dirname(BASE_DIR)

# Checkpoint finetune
CHECKPOINT  = os.path.join(ROOT_DIR, 'Finetune', 'outputs',
                           'progressive_s2_custom_ntut', 'seed_777', 'ft_best.pt')

# Backbones: Stream 2 (LLVIP)
RGB_BB_PATH = os.path.join(ROOT_DIR, 'backbones', 'llvip_rgb_best.pt')
THR_BB_PATH = os.path.join(ROOT_DIR, 'backbones', 'llvip_thermal_best.pt')

OUTPUT_ONNX = os.path.join(BASE_DIR, 'models', 'fusion_progressive_finetune.onnx')
IMG_SIZE    = 640

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =====================================================================
# MODEL -- giong het RGBTFusionDetector trong Finetune_CustomNTUT.ipynb
# =====================================================================
class RGBTFusionDetector(nn.Module):
    EXTRACT_LAYERS = {4: 64, 6: 128, 9: 256}

    def __init__(self, rgb_backbone, thermal_backbone, nc=1):
        super().__init__()
        self.rgb_stream     = rgb_backbone
        self.thermal_stream = thermal_backbone

        self.fuse_p3 = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.SiLU())
        self.fuse_p4 = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.SiLU())
        self.fuse_p5 = nn.Sequential(
            nn.Conv2d(512, 512, 1, bias=False), nn.BatchNorm2d(512), nn.SiLU())

        self.upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.td_c2f_p4  = C2f(512 + 256, 256, n=1, shortcut=False)
        self.td_c2f_p3  = C2f(256 + 128, 128, n=1, shortcut=False)
        self.bu_conv_p4 = Conv(128, 128, 3, 2)
        self.bu_c2f_p4  = C2f(128 + 256, 256, n=1, shortcut=False)
        self.bu_conv_p5 = Conv(256, 256, 3, 2)
        self.bu_c2f_p5  = C2f(256 + 512, 512, n=1, shortcut=False)
        self.detect     = Detect(nc=nc, ch=(128, 256, 512))
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

        p4_td  = self.td_c2f_p4(torch.cat([self.upsample(p5), p4], dim=1))
        p3_out = self.td_c2f_p3(torch.cat([self.upsample(p4_td), p3], dim=1))
        p4_out = self.bu_c2f_p4(torch.cat([self.bu_conv_p4(p3_out), p4_td], dim=1))
        p5_out = self.bu_c2f_p5(torch.cat([self.bu_conv_p5(p4_out), p5], dim=1))

        outs = self.detect([p3_out, p4_out, p5_out])
        if isinstance(outs, (list, tuple)):
            return torch.cat([o.flatten(2) for o in outs], dim=2)
        return outs


# =====================================================================
# EXPORT
# =====================================================================
if __name__ == '__main__':
    assert os.path.exists(CHECKPOINT), (
        f'Checkpoint not found: {CHECKPOINT}\n'
        f'scp root@<server>:/root/AIP491/Finetune/outputs/'
        f'progressive_s2_custom_ntut/seed_<best>/ft_best.pt {BASE_DIR}/'
    )

    print(f'Device: {device}')
    print(f'Loading backbones...')
    rgb_bb = nn.ModuleList(list(YOLO(RGB_BB_PATH).model.model)[:10]).to(device)
    thr_bb = nn.ModuleList(list(YOLO(THR_BB_PATH).model.model)[:10]).to(device)

    print(f'Building model...')
    model = RGBTFusionDetector(rgb_bb, thr_bb, nc=1).to(device)

    print(f'Loading checkpoint: {CHECKPOINT}')
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval()

    model.detect.export = True
    model.detect.format = 'onnx'

    dummy_rgb = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    dummy_thr = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    os.makedirs(os.path.dirname(OUTPUT_ONNX), exist_ok=True)
    print(f'Exporting -> {OUTPUT_ONNX}')

    torch.onnx.export(
        model,
        (dummy_rgb, dummy_thr),
        OUTPUT_ONNX,
        input_names=['rgb', 'thermal'],
        output_names=['output'],
        dynamic_axes={
            'rgb':     {0: 'batch'},
            'thermal': {0: 'batch'},
            'output':  {0: 'batch'},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    import onnx
    onnx.checker.check_model(onnx.load(OUTPUT_ONNX))
    size_mb = os.path.getsize(OUTPUT_ONNX) / (1024 * 1024)
    print(f'Export OK: {OUTPUT_ONNX} ({size_mb:.1f} MB)')

    # Huong dan thay the trong Demo
    print()
    print('Thay the model trong Demo:')
    print(f'  models/fusion_progressive_finetune.onnx  <- file nay')
    print(f'  Chinh app.py: ONNX_MODEL_PATH = "models/fusion_progressive_finetune.onnx"')
