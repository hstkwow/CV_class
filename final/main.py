"""命令行入口：数据下载、训练、评估、预测、报告。"""
import argparse
import csv
import json
import random
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from torchvision.transforms.functional import pil_to_tensor
from data import VOCDataset, CLASSES, collate
from model import build_model
from metrics import evaluate


def device_for(name):
    if name != 'auto': return torch.device(name)
    # torchvision 检测算子在 CPU/CUDA 上更稳妥；Mac 默认 CPU。
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loader(args, split, train=False):
    ds=VOCDataset(args.root,args.year,split,train,args.limit)
    return DataLoader(ds,batch_size=args.batch_size if train else 1,
                      shuffle=train,num_workers=args.workers,collate_fn=collate,
                      pin_memory=torch.cuda.is_available())


@torch.no_grad()
def validation(model, batches, device, eleven):
    model.eval(); predictions=[]; targets=[]
    # 评估保留低分框，避免展示阈值截断 PR 曲线。
    previous=model.roi_heads.score_thresh; model.roi_heads.score_thresh=.001
    try:
        for images, batch in batches:
            out=model([x.to(device) for x in images])
            predictions.extend([{k:v.cpu().numpy() for k,v in p.items()} for p in out])
            targets.extend([{k:v.numpy() for k,v in t.items()} for t in batch])
    finally: model.roi_heads.score_thresh=previous
    return evaluate(predictions,targets,CLASSES,eleven)


def load_checkpoint(path, device):
    # 只加载本项目生成、自己信任的检查点。
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    model=build_model(ckpt['image_size']);model.load_state_dict(ckpt['model'])
    return model.to(device),ckpt


def learning_rate_at(step, total_steps, base_lr, schedule, warmup_iters, steps_per_epoch, epochs):
    """先线性预热，再余弦或原有阶梯衰减；step 为零起点。"""
    warmup = min(warmup_iters, max(0, total_steps - 1))
    if step < warmup:
        return base_lr * (0.01 + 0.99 * (step + 1) / warmup)
    if schedule == 'cosine':
        progress = (step - warmup) / max(1, total_steps - warmup - 1)
        return base_lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))
    epoch = step // steps_per_epoch
    return base_lr * 0.3 ** (epoch // max(1, epochs // 3))


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = device_for(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用，请先检查驱动与 PyTorch；不要静默改用 CPU。')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if (out / 'history.csv').exists():
        raise FileExistsError('输出目录已有实验，请指定新的 --output，避免覆盖。')
    train_dl = loader(args, 'train', True); val_dl = loader(args, 'val')
    model = build_model(args.image_size).to(device)
    source_epoch = None
    if args.init_from:
        source, checkpoint = load_checkpoint(args.init_from, 'cpu')
        source_config = checkpoint.get('config', {})
        if source_config.get('year', args.year) != args.year:
            raise ValueError('初始化权重的数据年份不同；请明确使用同一数据划分。')
        model.load_state_dict(source.state_dict(), strict=True)
        source_epoch = checkpoint.get('epoch')
        del source, checkpoint
        print(f'从 {args.init_from} 的第 {source_epoch} 轮权重初始化；优化器重新开始。', flush=True)
    # eval() 总是固定到 image_size；仅 train() 随机选择这些短边尺度。
    if args.multiscale:
        model.transform.min_size = tuple(sorted(set(args.multiscale + [args.image_size])))
    model.transform.max_size = max(model.transform.min_size) * 2
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, weight_decay=5e-4)
    config = vars(args).copy()
    config.update(torch_version=str(torch.__version__), device_used=str(device),
        parameters=sum(p.numel() for p in model.parameters()),
        train_images=len(train_dl.dataset), val_images=len(val_dl.dataset), source_epoch=source_epoch)
    (out / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    total_steps = len(train_dl) * args.epochs
    best = -1
    # 先评估初始化权重，若后续训练退化则保留初始化模型为 best.pt。
    if args.init_from:
        metrics = validation(model, val_dl, device, args.voc07)
        best = metrics['mAP50']
        if best is None: raise ValueError('验证集没有可评估目标')
        metrics.update(split='val', year=args.year)
        torch.save({'model':model.state_dict(), 'image_size':args.image_size,
                    'epoch':0, 'metrics':metrics, 'config':config}, out / 'best.pt')
        (out / 'best_metrics.json').write_text(json.dumps(metrics, indent=2))
        print(f'继续训练前验证 mAP@0.5={best:.4f}（固定评估尺寸 {args.image_size}）', flush=True)
    with (out / 'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch','loss','mAP50','lr','seconds'])
        writer.writeheader()
        for epoch in range(args.epochs):
            model.train(); total = 0; started = time.monotonic()
            for step, (images, targets) in enumerate(train_dl):
                lr = learning_rate_at(epoch * len(train_dl) + step, total_steps,
                    args.lr, args.schedule, args.warmup_iters, len(train_dl), args.epochs)
                for group in optimizer.param_groups: group['lr'] = lr
                images = [x.to(device, non_blocking=True) for x in images]
                targets = [{k:v.to(device, non_blocking=True) for k,v in t.items()
                            if k in ('boxes','labels')} for t in targets]
                losses = model(images, targets); loss = sum(losses.values())
                if not torch.isfinite(loss): raise RuntimeError(f'非有限损失：{losses}')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step(); total += loss.item()
                if step % 20 == 0:
                    print(f'epoch {epoch+1}/{args.epochs}, step {step}/{len(train_dl)}, '
                          f'loss {loss.item():.4f}, lr {lr:.6f}', flush=True)
            metrics = validation(model, val_dl, device, args.voc07)
            metrics.update(split='val', year=args.year)
            score = metrics['mAP50']
            if score is None: raise ValueError('验证集没有可评估的非 difficult 目标')
            seconds = time.monotonic() - started
            writer.writerow(dict(epoch=epoch+1, loss=total/len(train_dl), mAP50=score,
                                 lr=lr, seconds=seconds)); f.flush()
            checkpoint = {'model':model.state_dict(), 'image_size':args.image_size,
                          'epoch':epoch+1, 'metrics':metrics, 'config':config}
            torch.save(checkpoint, out / 'last.pt')
            if score > best:
                best = score; torch.save(checkpoint, out / 'best.pt')
                (out / 'best_metrics.json').write_text(json.dumps(metrics, indent=2))
            print(f'验证 mAP@0.5={score:.4f}, 最佳={best:.4f}, 本轮耗时 {seconds:.0f}s', flush=True)


def run(args):
    if args.command=='download':
        from torchvision.datasets import VOCDetection
        VOCDetection(args.root,year=args.year,image_set=args.split,download=True)
    elif args.command=='train': train(args)
    elif args.command=='evaluate':
        dev=device_for(args.device);model,_=load_checkpoint(args.checkpoint,dev)
        result=validation(model,loader(args,args.split),dev,args.voc07)
        result.update(split=args.split,year=args.year,checkpoint=args.checkpoint)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    elif args.command=='predict':
        dev=device_for(args.device);model,_=load_checkpoint(args.checkpoint,dev);model.eval()
        image=Image.open(args.image).convert('RGB')
        with torch.no_grad(): pred=model([pil_to_tensor(image).float().to(dev)/255.])[0]
        draw=ImageDraw.Draw(image)
        for box,score,label in zip(pred['boxes'],pred['scores'],pred['labels']):
            if score.item()<args.threshold: continue
            coords=box.cpu().tolist();draw.rectangle(coords,outline='red',width=3)
            draw.text((coords[0],coords[1]),f'{CLASSES[label.item()-1]} {score.item():.2f}',fill='red')
        Path(args.output).parent.mkdir(parents=True,exist_ok=True);image.save(args.output)
    elif args.command=='report':
        from report import generate
        generate(Path(args.run), args.name, args.student_id)


def parser():
    p=argparse.ArgumentParser(description='机器视觉期末作业：自定义骨干的 VOC 目标检测')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ['download','train','evaluate','predict','report']:
        a=sub.add_parser(name)
        if name in ['download','train','evaluate']:
            a.add_argument('--root',default='data');a.add_argument('--year',choices=['2007','2012'],default='2007')
        if name in ['train','evaluate']:
            a.add_argument('--batch-size',type=int,default=2);a.add_argument('--workers',type=int,default=0)
            a.add_argument('--limit',type=int,default=0,help='仅调试使用；0 表示全部')
            a.add_argument('--voc07',action='store_true',help='使用 VOC2007 十一点 AP')
        if name in ['download','evaluate']: a.add_argument('--split',default='train',choices=['train','val','trainval','test'])
        if name in ['train','evaluate','predict']: a.add_argument('--device',default='auto')
        if name in ['evaluate','predict']: a.add_argument('--checkpoint',required=True)
        if name=='train':
            a.add_argument('--epochs',type=int,default=30);a.add_argument('--lr',type=float,default=.005)
            a.add_argument('--image-size',type=int,default=512);a.add_argument('--seed',type=int,default=42)
            a.add_argument('--output',default='runs/baseline')
            a.add_argument('--init-from', help='从已有模型权重开始新实验，不恢复优化器')
            a.add_argument('--schedule', choices=['step','cosine'], default='step')
            a.add_argument('--warmup-iters', type=int, default=0)
            a.add_argument('--multiscale', type=int, nargs='+', help='仅训练时随机短边尺度；均须不超过 image-size')
        if name=='evaluate': a.add_argument('--output',default='runs/evaluation.json')
        if name=='predict':
            a.add_argument('--image',required=True);a.add_argument('--threshold',type=float,default=.5)
            a.add_argument('--output',default='runs/prediction.jpg')
        if name=='report':
            a.add_argument('--run',default='runs/baseline')
            a.add_argument('--name',default=None)
            a.add_argument('--student-id',default=None)
    return p

if __name__=='__main__':
    args=parser().parse_args()
    if getattr(args,'epochs',1)<1 or getattr(args,'batch_size',1)<1 or getattr(args,'limit',0)<0:
        raise ValueError('epochs/batch-size 必须为正数，limit 不能为负数')
    if args.command == 'train':
        if args.lr <= 0 or args.image_size < 32 or args.warmup_iters < 0 or args.workers < 0:
            raise ValueError('lr 必须为正数；image-size >= 32；warmup-iters/workers >= 0')
        if args.multiscale and any(x < 32 or x > args.image_size for x in args.multiscale):
            raise ValueError('multiscale 各尺度须 >=32 且不超过 image-size（评估尺度）')
    run(args)
