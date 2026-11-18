# 机器视觉期末作业：PASCAL VOC 目标检测

本项目采用自行设计的轻量残差骨干与三尺度 FPN，并复用 torchvision Faster R-CNN 检测组件。没有使用现成预训练检测器或预训练权重。“自行设计”主要体现在骨干结构、通道、归一化及特征融合；RPN、RoIAlign、检测头与损失来自 torchvision，应在报告中如实说明。若教师要求全部检测组件自行实现，需要进一步扩展。

## 1. 环境

在终端进入本目录。建议 Python 3.10 或更新版本：

```bash
cd /Users/hstk/projects/code/CV_class/final
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python smoke_test.py
python -m unittest test_metrics.py
```

也可在已有 dl_env 环境安装。torch 与 torchvision 必须是官方兼容版本对；CUDA 用户按 https://pytorch.org/get-started/locally/ 安装适合本机的组合，再安装其余依赖。Mac 默认 CPU；完整训练建议 CUDA GPU。MPS 上部分检测算子可能不支持，不默认使用 MPS。

## 2. 数据准备

默认 VOC2007，train 训练、val 调参、test 最终评估。也支持 VOC2012 train/val，但 VOC2012 test 没有公开标注，不能用本项目直接本地评估 test。

```bash
python main.py download --root data --year 2007 --split train
python main.py download --root data --year 2007 --split test
```

若官方下载缓慢，可自行取得官方 tar 并解压到 data；目录应为：

```
data/VOCdevkit/VOC2007/
  JPEGImages/
  Annotations/
  ImageSets/Main/train.txt
  ImageSets/Main/val.txt
  ImageSets/Main/test.txt
```

不要把 CIFAR-10 当作本次数据集。VOC 包含 20 个检测类别，每张图可以有多个目标。数据读取将 VOC 的 1 起点坐标转换为 0 起点半开区间。训练忽略 difficult 目标；评估匹配 difficult 目标的预测不记为正例或负例。

## 3. 先做小规模验证

```bash
python main.py train --root data --limit 8 --epochs 1 --image-size 128 --batch-size 2 --output runs/debug --voc07
```

limit 仅供检查数据和流程，不代表正式实验。程序拒绝写入已有 history.csv 的目录，重复实验请换输出目录。

## 4. 正式训练和调参

```bash
python main.py train --root data --epochs 30 --image-size 512 --lr 0.005 --output runs/baseline --voc07
```

默认自动选 CUDA（可用时），否则 CPU。可用 `--device cpu` 显式指定。骨干从零训练；30 轮只是实验起点，不保证准确率。根据验证结果尝试更多轮数或学习率，保持测试集独立。例如：

```bash
python main.py train --root data --epochs 30 --lr 0.001 --output runs/lr001 --voc07
```

每轮写入 history.csv，并保存 last.pt、best.pt、best_metrics.json 和 config.json。best.pt 根据验证 mAP@0.5 选择。检查点保存模型与实验信息，不支持精确断点恢复优化器状态。

## 5. 最终测试与可视化

```bash
python main.py evaluate --root data --year 2007 --split test --checkpoint runs/baseline/best.pt --output runs/baseline/test_metrics.json --voc07
python main.py predict --checkpoint runs/baseline/best.pt --image data/VOCdevkit/VOC2007/JPEGImages/000001.jpg --output runs/baseline/prediction.jpg
python main.py report --run runs/baseline
```

示例图片名不存在时，替换为 JPEGImages 下的实际文件。预测展示阈值默认 0.5，可用 `--threshold 0.3` 调整。评估自动使用较低分数阈值 0.001，避免展示阈值截断 PR 曲线；最多保留 100 个预测/图片。评估算子在 CPU 上处理统计，兼容 CUDA 推理。

`--voc07` 使用 VOC2007 十一点 AP；不指定则使用精度包络积分 AP。比较实验时必须保持同一协议。这里只报告 IoU=0.5 的 mAP，不是 COCO mAP@[.5:.95]。缺少非 difficult 目标的类别不计入均值，完整数据评估更可靠。

## 6. 报告与提交

报告命令生成 `runs/baseline/实验报告.md` 与 curves.png，填入真实实验设置及验证 AP，分析部分需要结合你的曲线、误检图和对照实验完成。测试结果在 test_metrics.json，选定最终模型后将其加入报告，明确与验证结果区分。未训练前不提供虚构数字。

建议提交源代码、requirements.txt、README、实验报告、训练曲线、预测图片和必要模型权重；原始 VOC 数据不必打包。按文档要求将压缩包命名为“姓名_学号”。本项目不会自动发邮件。

## 网络设计

输入 → Conv(32,s2) → Conv(64,s2) → [Conv(128,s2)+残差] → [Conv(192,s2)+残差] → [Conv(256,s2)+残差]。

后三级为 C3/C4/C5，FPN 输出 128 通道 P3/P4/P5；每位置 6 个 anchors，尺度分别为 (32,64)、(128,192)、(256,384)，宽高比为 0.5/1/2。RPN 提供候选框，RoIAlign 取 7×7 特征，经分类和边框回归得到 20 类目标及背景。GroupNorm 避免小 batch 时 BatchNorm 统计不稳定。

优化目标为 RPN 分类+RPN 回归+RoI 分类+RoI 回归。SGD momentum=0.9，weight decay=0.0005，阶梯学习率衰减，梯度裁剪阈值 10。随机水平翻转、亮度和饱和度增强只用于训练。

## 参考

- https://docs.pytorch.org/vision/stable/models/faster_rcnn.html
- https://docs.pytorch.org/vision/stable/generated/torchvision.ops.FeaturePyramidNetwork.html
- https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.VOCDetection.html
- http://host.robots.ox.ac.uk/pascal/VOC/
