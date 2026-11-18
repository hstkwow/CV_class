"""VOC XML 读取；坐标转为零起点半开区间，保留 difficult 标记。"""
from pathlib import Path
import random
import xml.etree.ElementTree as ET
import torch
from PIL import Image, ImageOps, ImageEnhance
from torchvision.transforms.functional import pil_to_tensor

CLASSES = ('aeroplane','bicycle','bird','boat','bottle','bus','car','cat',
           'chair','cow','diningtable','dog','horse','motorbike','person',
           'pottedplant','sheep','sofa','train','tvmonitor')

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, year='2007', split='train', augment=False, limit=0):
        self.root = Path(root) / 'VOCdevkit' / ('VOC' + year)
        path = self.root / 'ImageSets/Main' / (split + '.txt')
        if not path.exists():
            raise FileNotFoundError(f'找不到 {path}；请先准备 VOC 数据集。')
        self.ids = [line.split()[0] for line in path.read_text().splitlines() if line.strip()]
        if limit: self.ids = self.ids[:limit]
        if not self.ids: raise ValueError('数据集为空')
        self.augment = augment
    def __len__(self): return len(self.ids)
    def __getitem__(self, index):
        name = self.ids[index]
        image = Image.open(self.root / 'JPEGImages' / (name + '.jpg')).convert('RGB')
        width, height = image.size
        boxes, labels, difficult = [], [], []
        xml = ET.parse(self.root / 'Annotations' / (name + '.xml')).getroot()
        for obj in xml.findall('object'):
            b = obj.find('bndbox')
            box = [max(0., float(b.findtext('xmin'))-1), max(0., float(b.findtext('ymin'))-1),
                   min(float(width), float(b.findtext('xmax'))), min(float(height), float(b.findtext('ymax')))]
            if box[2] <= box[0] or box[3] <= box[1]: continue
            hard = int(obj.findtext('difficult', '0'))
            if self.augment and hard: continue
            boxes.append(box); labels.append(CLASSES.index(obj.findtext('name'))+1); difficult.append(hard)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.augment:
            if random.random() < .5:
                image = ImageOps.mirror(image)
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            image = ImageEnhance.Color(image).enhance(random.uniform(.8, 1.2))
            image = ImageEnhance.Brightness(image).enhance(random.uniform(.8, 1.2))
        target = {'boxes': boxes, 'labels': torch.tensor(labels, dtype=torch.int64),
                  'difficult': torch.tensor(difficult, dtype=torch.bool),
                  'image_id': torch.tensor(index)}
        return pil_to_tensor(image).float()/255., target

def collate(batch): return tuple(zip(*batch))
