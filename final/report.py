"""从真实日志生成 Markdown 实验报告和曲线；不生成虚构结果。"""
import csv,json
from pathlib import Path

def generate(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    config=json.loads((run/'config.json').read_text())
    metrics=json.loads((run/'best_metrics.json').read_text())
    rows=list(csv.DictReader((run/'history.csv').open()))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axes,['loss','mAP50']):
        ax.plot([int(r['epoch']) for r in rows],[float(r[key]) for r in rows],marker='o')
        ax.set(xlabel='Epoch',ylabel=key);ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(run/'curves.png',dpi=160);plt.close(fig)
    ap='\n'.join(f'| {k} | {v:.4f} |' if v is not None else f'| {k} | 无非困难目标，未计入均值 |' for k,v in metrics['AP50'].items())
    text=f'''# 机器视觉技术期末实验报告

姓名：待填写　学号：待填写

## 任务与方法
使用 PASCAL VOC 的 20 个目标类别完成目标检测。设计轻量残差卷积骨干，使用 GroupNorm 和 SiLU，提取步长为 8、16、32 的三个尺度特征；FPN 将通道统一为 128，融合空间细节和语义信息。检测部分复用 torchvision Faster R-CNN 的 RPN、RoIAlign 和分类回归组件，非完整原创检测器，未使用预训练权重。

## 数据与实验设置
VOC{config['year']} 官方 train 用于训练、val 用于模型选择，未使用测试集调参。训练忽略 difficult 目标，评估按 VOC 规则忽略对 difficult 目标的匹配。训练增强为水平翻转和颜色/亮度扰动。训练图片 {config['train_images']} 张，验证图片 {config['val_images']} 张。

配置：学习率 {config['lr']}，batch size {config['batch_size']}，训练计划 {config['epochs']} 轮，当前记录 {len(rows)} 轮，输入短边 {config['image_size']}，随机种子 {config['seed']}。模型参数 {config['parameters']:,} 个，运行设备 {config['device_used']}。

## 实验结果
以下内容来自实际运行日志。评估指标为 IoU=0.5 的 {metrics['ap_protocol']} AP，非 COCO mAP@[.5:.95]。最佳验证 mAP@0.5 为 **{metrics['mAP50']:.4f}**。不含非困难目标的类别不计入均值。

{'注意：本次设置了 limit，仅为子集调试结果，不能作为完整 VOC 结果。' if config['limit'] else ''}

![训练曲线](curves.png)

| 类别 | AP@0.5 |
|---|---:|
{ap}

## 结果分析（请结合真实图片和曲线补充）
1. 描述损失与验证 AP 的变化，分析欠拟合或过拟合现象。
2. 比较高 AP 与低 AP 类别，检查小目标、遮挡、相似类别和漏检。
3. 将预测可视化图片加入报告，并区分高置信误检、定位偏差与漏检。
4. 建议以不同输出目录比较学习率或输入尺寸；保持相同数据划分和指标，不用测试集选择配置。

## 局限与改进
轻量骨干从零训练，可能需要较长训练；小目标和遮挡目标仍具挑战。可扩大训练数据、增强尺度变化或调整骨干，但需以实际对照实验支持改进结论。

## 复现与参考
项目 README 提供复现命令。参考 PyTorch/torchvision Faster R-CNN、FeaturePyramidNetwork 文档及 PASCAL VOC 评估协议；应如实注明第三方组件和工具辅助情况。
'''
    (run/'实验报告.md').write_text(text);print(run/'实验报告.md')
