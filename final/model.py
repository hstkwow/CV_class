"""自定义轻量骨干 + 三尺度 FPN + Faster R-CNN；无预训练权重。"""
from collections import OrderedDict
import torch
from torch import nn
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.GroupNorm(8, cout), nn.SiLU(inplace=True))


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(block(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels))
    def forward(self, x):
        return torch.nn.functional.silu(x + self.net(x))


class VisionBackbone(nn.Module):
    """下采样倍率 8/16/32；GroupNorm 适用于小批量检测训练。"""
    out_channels = 128
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(block(3, 32, 2), block(32, 64, 2))
        self.stage3 = nn.Sequential(block(64, 128, 2), Residual(128))
        self.stage4 = nn.Sequential(block(128, 192, 2), Residual(192))
        self.stage5 = nn.Sequential(block(192, 256, 2), Residual(256))
        self.fpn = FeaturePyramidNetwork([128, 192, 256], self.out_channels)
    def forward(self, x):
        c3 = self.stage3(self.stem(x))
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return self.fpn(OrderedDict([('0', c3), ('1', c4), ('2', c5)]))


def build_model(image_size=512):
    anchors = AnchorGenerator(((32, 64), (128, 192), (256, 384)),
                             ((0.5, 1., 2.),) * 3)
    return FasterRCNN(VisionBackbone(), num_classes=21,
        min_size=image_size, max_size=image_size * 2,
        rpn_anchor_generator=anchors,
        box_roi_pool=MultiScaleRoIAlign(['0', '1', '2'], 7, 2),
        box_detections_per_img=100)
