"""VOC AP@0.5：默认积分 AP；可选择 VOC2007 十一点 AP。无目标类别不计入均值。"""
import numpy as np

def iou(box, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    inter = np.maximum(0, np.minimum(box[2:], boxes[:, 2:])-np.maximum(box[:2], boxes[:, :2])).prod(1)
    area = np.maximum(0, boxes[:, 2:]-boxes[:, :2]).prod(1)
    a = np.maximum(0, np.asarray(box)[2:]-np.asarray(box)[:2]).prod()
    return inter / np.maximum(a+area-inter, 1e-12)

def average_precision(recall, precision, eleven=False):
    if eleven:
        return float(np.mean([precision[recall>=t].max() if np.any(recall>=t) else 0 for t in np.linspace(0,1,11)]))
    r = np.r_[0, recall, 1]; p = np.r_[0, precision, 0]
    p = np.maximum.accumulate(p[::-1])[::-1]
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx+1]-r[idx])*p[idx+1]))

def evaluate(predictions, targets, classes, eleven=False):
    aps = {}; counts = {}
    for cls, name in enumerate(classes, 1):
        gt = {}; det = []; positives = 0
        for image_id, (pred, target) in enumerate(zip(predictions, targets)):
            keep = np.asarray(target['labels']) == cls
            boxes = np.asarray(target['boxes']).reshape(-1,4)[keep]
            hard = np.asarray(target['difficult'], dtype=bool)[keep]
            gt[image_id] = (boxes, hard, np.zeros(len(boxes), dtype=bool))
            positives += int((~hard).sum())
            for b,s,l in zip(pred['boxes'],pred['scores'],pred['labels']):
                if int(l)==cls: det.append((float(s),image_id,np.asarray(b)))
        counts[name] = positives
        if not positives: aps[name] = None; continue
        det.sort(key=lambda d:d[0], reverse=True)
        tp=[];fp=[]
        for _, image_id, box in det:
            boxes,hard,matched = gt[image_id]
            overlaps = iou(box,boxes)
            j = int(overlaps.argmax()) if len(overlaps) else -1
            if j>=0 and overlaps[j]>=.5:
                if hard[j]: continue
                tp.append(int(not matched[j]));fp.append(int(matched[j]));matched[j]=True
            else: tp.append(0);fp.append(1)
        tp=np.cumsum(tp);fp=np.cumsum(fp)
        aps[name]=average_precision(tp/positives,tp/np.maximum(tp+fp,1),eleven)
    valid=[v for v in aps.values() if v is not None]
    return {'mAP50':float(np.mean(valid)) if valid else None,'AP50':aps,
            'positives':counts,'ap_protocol':'VOC2007-11point' if eleven else 'integral',
            'images':len(targets)}
