"""Ham dung chung cho tat ca cac script training.
Import module nay tu cac script trong Mid-stage/, Late-stage/, Early-stage/.
"""

import os, sys, gc, json, math, time, random, shutil
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.ops import box_iou, nms
from PIL import Image
from pathlib import Path
from collections import Counter

from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Conv, SPPF, Detect
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy

# ============================================================
# Config
# ============================================================
BASE_DIR = '/root/AIP491'
RGBT_DATA_DIR = os.path.join(BASE_DIR, 'data', 'RGBTDronePerson')
BACKBONES_DIR = os.path.join(BASE_DIR, 'backbones')

SDS_RGB_PATH  = os.path.join(BACKBONES_DIR, 'SDS_RGB_best.pt')
LLVIP_RGB_PATH = os.path.join(BACKBONES_DIR, 'llvip_rgb_best.pt')
LLVIP_THR_PATH = os.path.join(BACKBONES_DIR, 'llvip_thermal_best.pt')

# Stream 1: RGB=SDS, Thermal=LLVIP
# Stream 2: RGB=LLVIP, Thermal=LLVIP
STREAM_CONFIGS = {
    1: {'rgb': SDS_RGB_PATH,   'thr': LLVIP_THR_PATH, 'desc': 'SDS(RGB) + LLVIP(Thermal)'},
    2: {'rgb': LLVIP_RGB_PATH, 'thr': LLVIP_THR_PATH, 'desc': 'LLVIP(RGB) + LLVIP(Thermal)'},
}

CLASS_MAP = {'person': 0, 'rider': 0}
IMG_SIZE = 640
NUM_WORKERS = 4
GRAD_CLIP = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# VOC XML parser
# ============================================================
def parse_voc_xml(xml_path, class_map=CLASS_MAP):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip().lower()
        if name not in class_map:
            continue
        cls_id = class_map[name]
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        cx = (xmin + xmax) / 2.0 / w
        cy = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        boxes.append([cls_id, cx, cy, bw, bh])
    return boxes, w, h


# ============================================================
# Dataset setup: RGBTDronePerson VOC -> YOLO format
# ============================================================
def setup_mid_dataset(output_dir):
    """Tao YOLO dataset cho mid-fusion (chung labels, rieng images rgb/thermal).
    Tra ve duong dan FUSION_YOLO_DIR.
    """
    fusion_dir = os.path.join(output_dir, 'fusion_yolo')
    if os.path.isdir(os.path.join(fusion_dir, 'labels', 'train')):
        n = len(os.listdir(os.path.join(fusion_dir, 'labels', 'train')))
        if n > 0:
            print(f'Dataset da ton tai: {fusion_dir} ({n} labels)')
            return fusion_dir

    print('Dang tao YOLO dataset cho mid-fusion...')
    for sub in ['images/rgb/train', 'images/rgb/val',
                'images/thermal/train', 'images/thermal/val',
                'labels/train', 'labels/val']:
        os.makedirs(os.path.join(fusion_dir, sub), exist_ok=True)

    for split in ['train', 'val']:
        # Tim folder rgb: thu 'visible' truoc (RGBTDronePerson), roi 'rgb'
        rgb_src = None
        for rgb_name in ['visible', 'rgb']:
            p = os.path.join(RGBT_DATA_DIR, split, rgb_name)
            if os.path.isdir(p):
                rgb_src = p
                break
        if rgb_src is None:
            raise FileNotFoundError(f'RGB folder not found in {os.path.join(RGBT_DATA_DIR, split)}')

        thr_src = os.path.join(RGBT_DATA_DIR, split, 'thermal')
        ann_src = None
        for ann_name in ['annotation', 'annotations', 'Annotations']:
            p = os.path.join(RGBT_DATA_DIR, split, ann_name)
            if os.path.isdir(p):
                ann_src = p
                break
        if ann_src is None:
            raise FileNotFoundError(f'Annotation folder not found in {os.path.join(RGBT_DATA_DIR, split)}')

        count = 0
        for xml_file in sorted(os.listdir(ann_src)):
            if not xml_file.endswith('.xml'):
                continue
            stem = os.path.splitext(xml_file)[0]
            boxes, _, _ = parse_voc_xml(os.path.join(ann_src, xml_file))
            if not boxes:
                continue

            # Tim anh RGB
            rgb_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                p = os.path.join(rgb_src, stem + ext)
                if os.path.exists(p):
                    rgb_path = p
                    break
            # Tim anh Thermal
            thr_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                p = os.path.join(thr_src, stem + ext)
                if os.path.exists(p):
                    thr_path = p
                    break

            if not rgb_path or not thr_path:
                continue

            shutil.copy2(rgb_path, os.path.join(fusion_dir, 'images', 'rgb', split, f'{stem}.jpg'))
            shutil.copy2(thr_path, os.path.join(fusion_dir, 'images', 'thermal', split, f'{stem}.jpg'))

            with open(os.path.join(fusion_dir, 'labels', split, f'{stem}.txt'), 'w') as f:
                for box in boxes:
                    f.write(' '.join(f'{v:.6f}' if i > 0 else str(int(v)) for i, v in enumerate(box)) + '\n')
            count += 1

        print(f'  {split}: {count} mau')

    print(f'Dataset mid-fusion: {fusion_dir}')
    return fusion_dir


def setup_late_dataset(output_dir):
    """Tao 2 YOLO dataset rieng (rgb_yolo, thr_yolo) cho late-fusion.
    Tra ve (rgb_yolo_dir, thr_yolo_dir, rgb_yaml, thr_yaml).
    """
    rgb_dir = os.path.join(output_dir, 'rgb_yolo')
    thr_dir = os.path.join(output_dir, 'thr_yolo')

    check = os.path.join(rgb_dir, 'labels', 'train')
    if os.path.isdir(check) and len(os.listdir(check)) > 0:
        print(f'Dataset late da ton tai: {rgb_dir}')
    else:
        print('Dang tao YOLO dataset cho late-fusion...')
        for d in [rgb_dir, thr_dir]:
            for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
                os.makedirs(os.path.join(d, sub), exist_ok=True)

        for split in ['train', 'val']:
            rgb_src = None
            for rgb_name in ['visible', 'rgb']:
                p = os.path.join(RGBT_DATA_DIR, split, rgb_name)
                if os.path.isdir(p):
                    rgb_src = p
                    break
            if rgb_src is None:
                raise FileNotFoundError(f'RGB folder not found in {os.path.join(RGBT_DATA_DIR, split)}')

            thr_src = os.path.join(RGBT_DATA_DIR, split, 'thermal')
            ann_src = None
            for ann_name in ['annotation', 'annotations', 'Annotations']:
                p = os.path.join(RGBT_DATA_DIR, split, ann_name)
                if os.path.isdir(p):
                    ann_src = p
                    break
            if ann_src is None:
                raise FileNotFoundError(f'Annotation folder not found in {os.path.join(RGBT_DATA_DIR, split)}')

            count = 0
            for xml_file in sorted(os.listdir(ann_src)):
                if not xml_file.endswith('.xml'):
                    continue
                stem = os.path.splitext(xml_file)[0]
                boxes, _, _ = parse_voc_xml(os.path.join(ann_src, xml_file))
                if not boxes:
                    continue

                rgb_path = None
                for ext in ['.jpg', '.png', '.jpeg']:
                    p = os.path.join(rgb_src, stem + ext)
                    if os.path.exists(p):
                        rgb_path = p
                        break
                thr_path = None
                for ext in ['.jpg', '.png', '.jpeg']:
                    p = os.path.join(thr_src, stem + ext)
                    if os.path.exists(p):
                        thr_path = p
                        break

                if not rgb_path or not thr_path:
                    continue

                lbl_line = ''
                for box in boxes:
                    lbl_line += ' '.join(f'{v:.6f}' if i > 0 else str(int(v)) for i, v in enumerate(box)) + '\n'

                shutil.copy2(rgb_path, os.path.join(rgb_dir, 'images', split, f'{stem}.jpg'))
                shutil.copy2(thr_path, os.path.join(thr_dir, 'images', split, f'{stem}.jpg'))

                for d in [rgb_dir, thr_dir]:
                    with open(os.path.join(d, 'labels', split, f'{stem}.txt'), 'w') as f:
                        f.write(lbl_line)
                count += 1
            print(f'  {split}: {count} mau')

    # Tao YAML
    rgb_yaml = os.path.join(rgb_dir, 'dataset.yaml')
    thr_yaml = os.path.join(thr_dir, 'dataset.yaml')

    for yaml_path, data_dir, name in [(rgb_yaml, rgb_dir, 'RGB'), (thr_yaml, thr_dir, 'Thermal')]:
        with open(yaml_path, 'w') as f:
            f.write(f'path: {data_dir}\n')
            f.write('train: images/train\n')
            f.write('val: images/val\n')
            f.write('nc: 1\n')
            f.write("names: ['person']\n")

    return rgb_dir, thr_dir, rgb_yaml, thr_yaml


def setup_early_dataset(output_dir, llvip_dir=None):
    """Tao dataset cho early-fusion 4-channel (R,G,B,T).
    Cung cau truc voi setup_mid_dataset, dung chung RGBTDataset.
    Tra ve (llvip_ds_dir, rgbt_ds_dir).
    """
    llvip_ds_dir = os.path.join(output_dir, 'llvip_4ch')
    rgbt_ds_dir  = os.path.join(output_dir, 'rgbt_4ch')

    if llvip_dir is None:
        llvip_dir = os.path.join(BASE_DIR, 'data', 'LLVIP')

    # LLVIP: uu tien 'test' neu 'val' khong ton tai
    splits_llvip = ['train', 'val']
    if (not os.path.isdir(os.path.join(llvip_dir, 'visible', 'val')) and
            not os.path.isdir(os.path.join(llvip_dir, 'val', 'visible'))):
        if os.path.isdir(os.path.join(llvip_dir, 'visible', 'test')):
            splits_llvip = ['train', 'test']
            print('  LLVIP: dung split "test" cho validation')

    check_llvip = os.path.join(llvip_ds_dir, 'labels', 'train')
    if os.path.isdir(check_llvip) and len(os.listdir(check_llvip)) > 0:
        print(f'LLVIP 4ch dataset da ton tai: {llvip_ds_dir}')
    else:
        print('Dang tao LLVIP 4ch dataset...')
        _create_early_dataset(
            llvip_dir, llvip_ds_dir,
            rgb_subdir='visible', thr_subdir='infrared',
            ann_subdir='Annotations', splits=splits_llvip
        )

    check_rgbt = os.path.join(rgbt_ds_dir, 'labels', 'train')
    if os.path.isdir(check_rgbt) and len(os.listdir(check_rgbt)) > 0:
        print(f'RGBT 4ch dataset da ton tai: {rgbt_ds_dir}')
    else:
        print('Dang tao RGBTDronePerson 4ch dataset...')
        _create_early_dataset(
            RGBT_DATA_DIR, rgbt_ds_dir,
            rgb_subdir='visible', thr_subdir='thermal',
            ann_subdir='annotation', splits=['train', 'val']
        )

    return llvip_ds_dir, rgbt_ds_dir


def _create_early_dataset(src_dir, dst_dir, rgb_subdir, thr_subdir, ann_subdir, splits):
    """Tao dataset voi cau truc images/rgb/ + images/thermal/ + labels/.
    Ho tro ca 2 layout: src/split/subdir va src/subdir/split (LLVIP).
    """
    for sub in ['images/rgb/train', 'images/rgb/val',
                'images/thermal/train', 'images/thermal/val',
                'labels/train', 'labels/val']:
        os.makedirs(os.path.join(dst_dir, sub), exist_ok=True)

    for split in splits:
        # Layout 1: src_dir/split/subdir
        rgb_src = os.path.join(src_dir, split, rgb_subdir)
        thr_src = os.path.join(src_dir, split, thr_subdir)
        ann_src = os.path.join(src_dir, split, ann_subdir)

        # Fallback Layout 2: src_dir/subdir/split (LLVIP)
        if not os.path.isdir(rgb_src):
            rgb_src = os.path.join(src_dir, rgb_subdir, split)
            thr_src = os.path.join(src_dir, thr_subdir, split)
            ann_src = os.path.join(src_dir, ann_subdir)

        # Fallback annotation folder name
        if not os.path.isdir(ann_src):
            for alt in [ann_subdir, 'Annotations', 'annotations', 'annotation']:
                for base in [os.path.join(src_dir, split), src_dir]:
                    p = os.path.join(base, alt)
                    if os.path.isdir(p):
                        ann_src = p
                        break
                if os.path.isdir(ann_src):
                    break

        if not os.path.isdir(ann_src):
            print(f'  Warning: annotation folder not found for split={split}, skip.')
            continue

        # Chuan hoa ten split cho output folder (test -> val)
        out_split = 'val' if split == 'test' else split

        count = 0
        for xml_file in sorted(os.listdir(ann_src)):
            if not xml_file.endswith('.xml'):
                continue
            stem = os.path.splitext(xml_file)[0]
            boxes, _, _ = parse_voc_xml(os.path.join(ann_src, xml_file))
            if not boxes:
                continue

            rgb_path = thr_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                p = os.path.join(rgb_src, stem + ext)
                if os.path.exists(p):
                    rgb_path = p
                    break
            for ext in ['.jpg', '.png', '.jpeg']:
                p = os.path.join(thr_src, stem + ext)
                if os.path.exists(p):
                    thr_path = p
                    break

            if not rgb_path or not thr_path:
                continue

            shutil.copy2(rgb_path, os.path.join(dst_dir, 'images', 'rgb', out_split, f'{stem}.jpg'))
            shutil.copy2(thr_path, os.path.join(dst_dir, 'images', 'thermal', out_split, f'{stem}.jpg'))

            with open(os.path.join(dst_dir, 'labels', out_split, f'{stem}.txt'), 'w') as f:
                for box in boxes:
                    f.write(' '.join(f'{v:.6f}' if i > 0 else str(int(v)) for i, v in enumerate(box)) + '\n')
            count += 1

        print(f'  {split}: {count} mau')


# ============================================================
# RGBTDataset cho mid-fusion
# ============================================================
class RGBTDataset(Dataset):
    def __init__(self, fusion_yolo_dir, split='train', img_size=640,
                 blur_prob=0.5, blur_kernels=(3, 5, 7), blur_sigma=(0.5, 2.0),
                 flip_prob=0.5):
        self.img_size = img_size
        self.is_train = (split == 'train')
        self.blur_prob = blur_prob if self.is_train else 0.0
        self.blur_kernels = blur_kernels
        self.blur_sigma = blur_sigma
        self.flip_prob = flip_prob if self.is_train else 0.0

        self.rgb_dir = Path(fusion_yolo_dir) / 'images' / 'rgb' / split
        self.thr_dir = Path(fusion_yolo_dir) / 'images' / 'thermal' / split
        self.lbl_dir = Path(fusion_yolo_dir) / 'labels' / split

        rgb_stems = {p.stem for p in self.rgb_dir.glob('*.jpg')}
        thr_stems = {p.stem for p in self.thr_dir.glob('*.jpg')}
        lbl_stems = {p.stem for p in self.lbl_dir.glob('*.txt')}
        self.stems = sorted(rgb_stems & thr_stems & lbl_stems)

        self.resize = T.Resize((img_size, img_size))
        self.to_tensor = T.ToTensor()

        aug_info = []
        if self.blur_prob > 0:
            aug_info.append(f'blur(p={self.blur_prob}, k={self.blur_kernels})')
        if self.flip_prob > 0:
            aug_info.append(f'hflip(p={self.flip_prob})')
        aug_str = ', '.join(aug_info) if aug_info else 'none'
        print(f'RGBTDataset [{split}]: {len(self.stems)} mau | augment: {aug_str}')

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        rgb_img = Image.open(self.rgb_dir / f'{stem}.jpg').convert('RGB')
        thr_img = Image.open(self.thr_dir / f'{stem}.jpg').convert('L').convert('RGB')
        rgb_img = self.resize(rgb_img)
        thr_img = self.resize(thr_img)

        if random.random() < self.blur_prob:
            k = random.choice(self.blur_kernels)
            sigma = random.uniform(self.blur_sigma[0], self.blur_sigma[1])
            rgb_img = TF.gaussian_blur(rgb_img, kernel_size=[k, k], sigma=[sigma, sigma])

        labels = []
        with open(self.lbl_dir / f'{stem}.txt') as f:
            for line in f:
                v = list(map(float, line.strip().split()))
                if len(v) == 5:
                    labels.append(v)

        if random.random() < self.flip_prob:
            rgb_img = TF.hflip(rgb_img)
            thr_img = TF.hflip(thr_img)
            labels = [[v[0], 1.0 - v[1], v[2], v[3], v[4]] for v in labels]

        rgb = self.to_tensor(rgb_img)
        thr = self.to_tensor(thr_img)
        lbl_t = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))
        return rgb, thr, lbl_t, stem


def collate_fn(batch):
    rgbs, thrs, labels_list, stems = zip(*batch)
    rgbs = torch.stack(rgbs)
    thrs = torch.stack(thrs)
    out = []
    for i, lbl in enumerate(labels_list):
        if lbl.shape[0] > 0:
            out.append(torch.cat([torch.full((len(lbl), 1), i), lbl], dim=1))
    batch_labels = torch.cat(out, dim=0) if out else torch.zeros((0, 6))
    return rgbs, thrs, batch_labels, stems


# ============================================================
# EarlyFusion4chDetector: YOLOv8 4-channel (R,G,B,T)
# ============================================================
class EarlyFusion4chDetector(nn.Module):
    """YOLOv8 voi first conv sua thanh 4-channel: [R, G, B, T].
    T la kenh thermal grayscale (1 kenh).
    Forward nhan (rgb [B,3,H,W], thermal [B,3,H,W]) tu RGBTDataset,
    tu dong lay kenh dau cua thermal (L->RGB nen 3 kenh giong nhau).
    Tuong hop voi LossModel, run_epoch, evaluate_model.
    """
    def __init__(self, base_model_path, nc=1):
        super().__init__()
        yolo = YOLO(base_model_path)
        self._yolo_model = yolo.model

        # Sua first conv: Conv2d(3, out_ch, ...) -> Conv2d(4, out_ch, ...)
        first_layer = self._yolo_model.model[0]  # ultralytics Conv block
        old_conv = first_layer.conv               # nn.Conv2d(3, ...)

        new_conv = nn.Conv2d(
            4, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )
        with torch.no_grad():
            # Channels 0-2: copy tu pretrained RGB weights
            new_conv.weight[:, :3, :, :] = old_conv.weight.clone()
            # Channel 3 (thermal): khoi tao bang mean cua 3 kenh RGB
            new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        first_layer.conv = new_conv

        # Can thiet cho LossModel
        self.detect = self._yolo_model.model[-1]

    def forward(self, rgb, thermal):
        # thermal: [B, 3, H, W] (L->RGB trong RGBTDataset, 3 kenh giong nhau)
        # Chi lay kenh dau lam kenh T: [B, 1, H, W]
        t = thermal[:, :1, :, :]
        x = torch.cat([rgb, t], dim=1)  # [B, 4, H, W]
        return self._yolo_model(x)


# ============================================================
# Load YOLOv8n backbone (layers 0-9)
# ============================================================
def load_yolov8n_backbone(backbone_path):
    if backbone_path and os.path.exists(backbone_path):
        try:
            yolo = YOLO(backbone_path)
            bb = nn.ModuleList(list(yolo.model.model)[:10])
            return bb.to(device)
        except Exception as e:
            print(f'Warning: load {backbone_path} that bai ({e}), dung yolov8n.pt')
    base = YOLO('yolov8n.pt')
    return nn.ModuleList(list(base.model.model)[:10]).to(device)


# ============================================================
# Loss wrapper cho v8DetectionLoss
# ============================================================
class LossModel(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.model = nn.ModuleList([detector.detect])
        self.nc    = detector.detect.nc
        self.args  = type('a', (), {
            'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
            'pose': 12.0, 'kobj': 1.0
        })()


# ============================================================
# Training epoch
# ============================================================
def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch_idx, (rgb, thr, labels, _) in enumerate(loader):
            rgb    = rgb.to(device)
            thr    = thr.to(device)
            labels = labels.to(device)

            preds = model(rgb, thr)
            batch_size = rgb.shape[0]

            batch_dict = {
                'cls':       labels[:, 1:2],
                'bboxes':    labels[:, 2:6],
                'batch_idx': labels[:, 0],
                'img':       rgb,
            }
            loss_result = criterion(preds, batch_dict)
            loss = loss_result[0].sum() if isinstance(loss_result, (tuple, list)) else loss_result.sum()
            loss = loss / batch_size

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
                )
                optimizer.step()

            total_loss += loss.item()
            n += 1

            if train and batch_idx % 50 == 0:
                print(f'    [{batch_idx:4d}/{len(loader)}] loss={loss.item():.4f}')

    return total_loss / max(n, 1)


# ============================================================
# NMS + COCO AP Evaluation
# ============================================================
def custom_nms(prediction, conf_thres=0.25, iou_thres=0.45, nc=1):
    if prediction.dim() == 3 and prediction.shape[1] < prediction.shape[2]:
        prediction = prediction.transpose(1, 2)
    output = []
    for xi in range(prediction.shape[0]):
        x = prediction[xi]
        box = x[:, :4]
        cls_scores = x[:, 4:4+nc]
        conf, cls_idx = cls_scores.max(dim=1)
        mask = conf >= conf_thres
        box, conf, cls_idx = box[mask], conf[mask], cls_idx[mask]
        if len(box) == 0:
            output.append(torch.zeros((0, 6), device=prediction.device))
            continue
        xyxy = xywh2xyxy(box)
        keep = nms(xyxy, conf, iou_thres)
        output.append(torch.cat([xyxy[keep], conf[keep].unsqueeze(1), cls_idx[keep].float().unsqueeze(1)], dim=1))
    return output


def compute_ap_101(precision_curve, recall_curve):
    mrec = np.concatenate(([0.0], recall_curve, [1.0]))
    mpre = np.concatenate(([1.0], precision_curve, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    recall_points = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_points:
        indices = np.where(mrec >= r)[0]
        if len(indices) > 0:
            ap += mpre[indices[0]]
    return ap / 101.0


def evaluate_model(model, val_loader, img_size=640, nms_iou=0.45):
    model.eval()
    all_pred_boxes, all_pred_scores, all_gt_boxes, all_img_ids = [], [], [], []
    n_gt_total = 0
    img_id = 0

    with torch.no_grad():
        for rgb, thr, labels, _ in val_loader:
            rgb = rgb.to(device)
            thr = thr.to(device)
            preds_raw = model(rgb, thr)
            pred_t = preds_raw[0] if isinstance(preds_raw, (list, tuple)) else preds_raw
            dets_list = custom_nms(pred_t, conf_thres=0.001, iou_thres=nms_iou, nc=1)

            for img_i, dets in enumerate(dets_list):
                gt_mask = labels[:, 0] == img_i
                gt = labels[gt_mask]
                n_gt = len(gt)
                gt_xyxy = xywh2xyxy(gt[:, 2:6]) * img_size if n_gt > 0 else torch.zeros((0, 4))
                all_gt_boxes.append(gt_xyxy)
                n_gt_total += n_gt

                if len(dets) > 0:
                    for di in range(len(dets)):
                        all_pred_boxes.append(dets[di, :4].cpu())
                        all_pred_scores.append(float(dets[di, 4].cpu()))
                        all_img_ids.append(img_id)
                img_id += 1

    all_pred_scores_np = np.array(all_pred_scores)
    all_img_ids_np = np.array(all_img_ids)
    sorted_indices = np.argsort(-all_pred_scores_np)

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    ap_per_iou = {}

    for iou_thresh in iou_thresholds:
        matched_gt = {i: set() for i in range(len(all_gt_boxes))}
        tp_arr = np.zeros(len(sorted_indices))
        fp_arr = np.zeros(len(sorted_indices))

        for rank, det_idx in enumerate(sorted_indices):
            img_id_val = all_img_ids_np[det_idx]
            pred_box = all_pred_boxes[det_idx]
            gt_boxes = all_gt_boxes[img_id_val]
            if len(gt_boxes) == 0:
                fp_arr[rank] = 1
                continue
            ious = box_iou(pred_box.unsqueeze(0), gt_boxes)[0]
            max_iou = ious.max().item()
            max_gt_idx = ious.argmax().item()
            if max_iou >= iou_thresh and max_gt_idx not in matched_gt[img_id_val]:
                tp_arr[rank] = 1
                matched_gt[img_id_val].add(max_gt_idx)
            else:
                fp_arr[rank] = 1

        cum_tp = np.cumsum(tp_arr)
        cum_fp = np.cumsum(fp_arr)
        prec_curve = cum_tp / (cum_tp + cum_fp + 1e-6)
        rec_curve = cum_tp / (n_gt_total + 1e-6)
        ap = compute_ap_101(prec_curve, rec_curve)
        ap_per_iou[f'{iou_thresh:.2f}'] = ap

    map50 = ap_per_iou.get('0.50', 0.0)
    map50_95 = np.mean(list(ap_per_iou.values()))

    # P/R/F1 tai conf>=0.25, IoU>=0.5
    conf_mask = all_pred_scores_np[sorted_indices] >= 0.25
    tp_025, fp_025 = 0, 0
    matched_025 = {i: set() for i in range(len(all_gt_boxes))}
    for rank in range(len(sorted_indices)):
        if not conf_mask[rank]:
            break
        det_idx = sorted_indices[rank]
        img_id_val = all_img_ids_np[det_idx]
        pred_box = all_pred_boxes[det_idx]
        gt_boxes = all_gt_boxes[img_id_val]
        if len(gt_boxes) == 0:
            fp_025 += 1
            continue
        ious = box_iou(pred_box.unsqueeze(0), gt_boxes)[0]
        max_iou = ious.max().item()
        max_gt_idx = ious.argmax().item()
        if max_iou >= 0.5 and max_gt_idx not in matched_025[img_id_val]:
            tp_025 += 1
            matched_025[img_id_val].add(max_gt_idx)
        else:
            fp_025 += 1

    fn_025 = n_gt_total - tp_025
    precision = tp_025 / (tp_025 + fp_025 + 1e-9)
    recall = tp_025 / (tp_025 + fn_025 + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'map50': map50, 'map50_95': map50_95, 'ap_per_iou': ap_per_iou
    }


# ============================================================
# Plot loss curves
# ============================================================
def plot_loss_curves(all_histories, seeds, save_path, title=''):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes

    for i, seed in enumerate(seeds):
        if seed not in all_histories:
            continue
        h = all_histories[seed]
        ep = list(range(1, len(h['train_loss']) + 1))
        ax1.plot(ep, h['train_loss'], color=colors[i], linestyle='-',  alpha=0.7, label=f'Train s{seed}')
        ax1.plot(ep, h['val_loss'],   color=colors[i], linestyle='--', alpha=0.7, label=f'Val s{seed}')
        if 'lr' in h:
            ax2.plot(ep, h['lr'], color=colors[i], label=f'Seed {seed}')

    ax1.set(title='Loss', xlabel='Epoch', ylabel='Loss')
    ax1.legend(fontsize=7); ax1.grid(True)
    ax2.set(title='Learning Rate', xlabel='Epoch', ylabel='LR')
    ax2.legend(); ax2.grid(True)

    if title:
        plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'Loss curves saved: {save_path}')
    plt.close()


# ============================================================
# Print summary table
# ============================================================
def print_summary(all_results, seeds, title=''):
    print(f'\n{"="*70}')
    if title:
        print(f'  {title}')
        print(f'{"="*70}')
    print(f'{"Seed":<8} {"Precision":>10} {"Recall":>10} {"F1":>10} {"mAP@0.5":>10} {"mAP@.5:.95":>14}')
    print(f'{"-"*62}')

    for seed in seeds:
        if seed in all_results:
            m = all_results[seed]
            print(f' {seed:<7} {m["precision"]:>10.4f} {m["recall"]:>10.4f} {m["f1"]:>10.4f} '
                  f'{m["map50"]:>10.4f} {m["map50_95"]:>14.4f}')

    if len(all_results) > 1:
        print(f'{"-"*62}')
        for metric_name, key in [('Precision', 'precision'), ('Recall', 'recall'), ('F1', 'f1'),
                                  ('mAP@0.5', 'map50'), ('mAP@.5:.95', 'map50_95')]:
            vals = [all_results[s][key] for s in seeds if s in all_results]
            print(f'  {metric_name:<14}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')
    print(f'{"="*70}')


# ============================================================
# WBF cho late-fusion
# ============================================================
def late_fusion_wbf(rgb_model, thr_model, rgb_img, thr_img, img_size=640,
                    conf_thres=0.001, iou_thres=0.55, weights=None):
    """Chay 2 model, gop ket qua bang Weighted Boxes Fusion."""
    from ensemble_boxes import weighted_boxes_fusion

    if weights is None:
        weights = [1.0, 1.0]

    rgb_results = rgb_model.predict(rgb_img, imgsz=img_size, conf=conf_thres, verbose=False)
    thr_results = thr_model.predict(thr_img, imgsz=img_size, conf=conf_thres, verbose=False)

    all_boxes, all_scores, all_labels = [], [], []

    for results in [rgb_results, thr_results]:
        boxes_norm, scores, labels = [], [], []
        if len(results) > 0 and results[0].boxes is not None:
            r = results[0]
            h_img, w_img = r.orig_shape
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                boxes_norm.append([x1/w_img, y1/h_img, x2/w_img, y2/h_img])
                scores.append(float(b.conf.cpu()))
                labels.append(0)
        all_boxes.append(boxes_norm if boxes_norm else np.zeros((0, 4)).tolist())
        all_scores.append(scores if scores else [])
        all_labels.append(labels if labels else [])

    if all(len(s) == 0 for s in all_scores):
        return np.zeros((0, 4)), np.array([]), np.array([])

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=weights, iou_thr=iou_thres, skip_box_thr=conf_thres
    )
    return fused_boxes, fused_scores, fused_labels
