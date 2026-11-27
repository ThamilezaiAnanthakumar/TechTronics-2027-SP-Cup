"""
RetinaFace face detector (MobileNet-0.25 backbone), the Python counterpart of MATLAB's
faceDetector("small-network") used in src/face/detect_faces.m.

Network: Deng et al., "RetinaFace: Single-stage Dense Face Localisation in the Wild", CVPR 2020.
Architecture and inference steps follow the reference PyTorch implementation
(github.com/biubug6/Pytorch_Retinaface, MIT licence; as packaged in batch-face, MIT licence).
Weights: mobilenet0.25_Final.pth (WIDER FACE), 1.8 MB, downloaded once to models/retinaface/
(gitignored) and loaded with torch.load(weights_only=True), so the file cannot run code.

MathWorks' trained model cannot be loaded in Python, so boxes can differ slightly from MATLAB;
the detector type, backbone and settings (threshold 0.5, MinSize 10 px, CPU) are the same.
"""

import hashlib
import urllib.request
from itertools import product
from math import ceil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter

REPO = Path(__file__).resolve().parents[3]
WEIGHTS = REPO / "models" / "retinaface" / "mobilenet0.25_Final.pth"
WEIGHTS_URL = "https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth"
WEIGHTS_SHA256 = "2979b33ffafda5d74b6948cd7a5b9a7a62f62b949cef24e95fd15d2883a65220"

CFG = dict(min_sizes=[[16, 32], [64, 128], [256, 512]], steps=[8, 16, 32], variance=[0.1, 0.2])
BGR_MEAN = (104.0, 117.0, 123.0)
NMS_IOU = 0.4

torch.set_grad_enabled(False)


# --------------------------------------------------------------- network

def _conv_bn(inp, oup, stride=1, leaky=0.0):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup),
                         nn.LeakyReLU(negative_slope=leaky, inplace=True))


def _conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup))


def _conv_bn1x1(inp, oup, stride, leaky=0.0):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, stride, 0, bias=False), nn.BatchNorm2d(oup),
                         nn.LeakyReLU(negative_slope=leaky, inplace=True))


def _conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True))


class _SSH(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        leaky = 0.1 if cout <= 64 else 0.0
        self.conv3X3 = _conv_bn_no_relu(cin, cout // 2, 1)
        self.conv5X5_1 = _conv_bn(cin, cout // 4, 1, leaky)
        self.conv5X5_2 = _conv_bn_no_relu(cout // 4, cout // 4, 1)
        self.conv7X7_2 = _conv_bn(cout // 4, cout // 4, 1, leaky)
        self.conv7x7_3 = _conv_bn_no_relu(cout // 4, cout // 4, 1)

    def forward(self, x):
        c5_1 = self.conv5X5_1(x)
        return F.relu(torch.cat([self.conv3X3(x), self.conv5X5_2(c5_1),
                                 self.conv7x7_3(self.conv7X7_2(c5_1))], dim=1))


class _FPN(nn.Module):
    def __init__(self, cins, cout):
        super().__init__()
        leaky = 0.1 if cout <= 64 else 0.0
        self.output1 = _conv_bn1x1(cins[0], cout, 1, leaky)
        self.output2 = _conv_bn1x1(cins[1], cout, 1, leaky)
        self.output3 = _conv_bn1x1(cins[2], cout, 1, leaky)
        self.merge1 = _conv_bn(cout, cout, leaky=leaky)
        self.merge2 = _conv_bn(cout, cout, leaky=leaky)

    def forward(self, feats):
        f = list(feats.values())
        o1, o2, o3 = self.output1(f[0]), self.output2(f[1]), self.output3(f[2])
        o2 = self.merge2(o2 + F.interpolate(o3, size=o2.shape[2:], mode="nearest"))
        o1 = self.merge1(o1 + F.interpolate(o2, size=o1.shape[2:], mode="nearest"))
        return [o1, o2, o3]


class _MobileNetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(_conv_bn(3, 8, 2, 0.1), _conv_dw(8, 16, 1), _conv_dw(16, 32, 2),
                                    _conv_dw(32, 32, 1), _conv_dw(32, 64, 2), _conv_dw(64, 64, 1))
        self.stage2 = nn.Sequential(_conv_dw(64, 128, 2), *[_conv_dw(128, 128, 1) for _ in range(5)])
        self.stage3 = nn.Sequential(_conv_dw(128, 256, 2), _conv_dw(256, 256, 1))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)


class _Head(nn.Module):
    def __init__(self, cin, nout):
        super().__init__()
        self.conv1x1 = nn.Conv2d(cin, nout, 1, 1, 0)

    def forward(self, x):
        return self.conv1x1(x).permute(0, 2, 3, 1).contiguous()


class _RetinaFaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = IntermediateLayerGetter(_MobileNetV1(), {"stage1": 1, "stage2": 2, "stage3": 3})
        self.fpn = _FPN([64, 128, 256], 64)
        self.ssh1, self.ssh2, self.ssh3 = _SSH(64, 64), _SSH(64, 64), _SSH(64, 64)
        self.ClassHead = nn.ModuleList([_Head(64, 2 * 2) for _ in range(3)])
        self.BboxHead = nn.ModuleList([_Head(64, 2 * 4) for _ in range(3)])
        self.LandmarkHead = nn.ModuleList([_Head(64, 2 * 10) for _ in range(3)])

    def forward(self, x):
        fpn = self.fpn(self.body(x))
        feats = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]
        n = x.shape[0]
        loc = torch.cat([self.BboxHead[i](f).view(n, -1, 4) for i, f in enumerate(feats)], 1)
        conf = torch.cat([self.ClassHead[i](f).view(n, -1, 2) for i, f in enumerate(feats)], 1)
        return loc, F.softmax(conf, dim=-1)


# --------------------------------------------------------------- inference helpers

def _priors(h, w):
    anchors = []
    for k, step in enumerate(CFG["steps"]):
        fh, fw = ceil(h / step), ceil(w / step)
        for i, j in product(range(fh), range(fw)):
            for ms in CFG["min_sizes"][k]:
                anchors += [(j + 0.5) * step / w, (i + 0.5) * step / h, ms / w, ms / h]
    return torch.tensor(anchors, dtype=torch.float32).view(-1, 4)


def _nms(dets, thresh):
    x1, y1, x2, y2, s = dets.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = s.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        order = order[1:][inter / (areas[i] + areas[order[1:]] - inter) <= thresh]
    return keep


def ensure_weights(path=WEIGHTS):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading RetinaFace weights -> {path}")
        urllib.request.urlretrieve(WEIGHTS_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise RuntimeError(f"{path}: unexpected SHA-256 {digest}")
    return path


class RetinaFaceDetector:
    """detect(rgb, roi=None) -> (boxes Mx4 [x y w h] in frame pixels, 0-based; scores M)."""

    def __init__(self, threshold=0.5, min_size=(10, 10), weights=WEIGHTS, threads=None):
        if threads:
            torch.set_num_threads(threads)
        self.threshold = float(threshold)
        self.min_h, self.min_w = (float(min_size[0]), float(min_size[1]))
        self.net = _RetinaFaceNet()
        sd = torch.load(ensure_weights(weights), map_location="cpu", weights_only=True)
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        self.net.load_state_dict(sd, strict=True)   # every tensor must match the architecture
        self.net.eval()
        self._prior_cache = {}

    def detect(self, rgb, roi=None):
        """rgb HxWx3 uint8. roi [x y w h] (0-based, integer) restricts detection to that window."""
        ox = oy = 0
        img = rgb
        if roi is not None:
            x, y, w, h = (int(round(v)) for v in roi)
            img = rgb[y:y + h, x:x + w]
            ox, oy = x, y
            if img.shape[0] < 16 or img.shape[1] < 16:
                return np.zeros((0, 4)), np.zeros(0)
        h, w = img.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1], dtype=np.float32))  # RGB -> BGR
        x = (x - torch.tensor(BGR_MEAN)).permute(2, 0, 1).unsqueeze(0)
        loc, conf = self.net(x)
        key = (h, w)
        if key not in self._prior_cache:
            self._prior_cache[key] = _priors(h, w)
        pr = self._prior_cache[key]
        v = CFG["variance"]
        b = torch.cat((pr[:, :2] + loc[0, :, :2] * v[0] * pr[:, 2:], pr[:, 2:] * torch.exp(loc[0, :, 2:] * v[1])), 1)
        b[:, :2] -= b[:, 2:] / 2
        b[:, 2:] += b[:, :2]
        b = (b * torch.tensor([w, h, w, h], dtype=torch.float32)).numpy()
        s = conf[0, :, 1].numpy()
        keep = s > self.threshold
        b, s = b[keep], s[keep]
        if len(s) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        order = s.argsort()[::-1][:5000]
        dets = np.hstack([b[order], s[order, None]]).astype(np.float32)
        dets = dets[_nms(dets, NMS_IOU)][:750]
        boxes = np.column_stack([dets[:, 0] + ox, dets[:, 1] + oy,
                                 dets[:, 2] - dets[:, 0], dets[:, 3] - dets[:, 1]]).astype(float)
        scores = dets[:, 4].astype(float)
        ok = (boxes[:, 2] >= self.min_w) & (boxes[:, 3] >= self.min_h)
        return boxes[ok], scores[ok]
