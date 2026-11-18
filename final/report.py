"""从真实日志生成中文报告；区分验证/测试，保留已有姓名与学号。"""
import csv
import json
import re
from pathlib import Path


def ap_table(metrics):
    return '\n'.join(f'| {k} | {v:.4f} |' if v is not None else
                     f'| {k} | 无非困难目标，未计入均值 |'
                     for k, v in metrics['AP50'].items())


def generate(run, name=None, student_id=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    metrics = json.loads((run / 'best_metrics.json').read_text())
    with (run / 'history.csv').open() as f:
        rows = list(csv.DictReader(f))
    if not rows: raise ValueError('history.csv 尚无已完成轮次，请完成至少一轮训练')
    path = run / '实验报告.md'
    old = path.read_text() if path.exists() else ''
    identity = re.search(r'姓名[：:]\s*([^\s]+)\s+学号[：:]\s*([^\s]+)', old)
    name = name or (identity[1] if identity else '待填写')
    student_id = student_id or (identity[2] if identity else '待填写')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ['loss', 'mAP50']):
        ax.plot([int(r['epoch']) for r in rows], [float(r[key]) for r in rows], marker='o')
        ax.set(xlabel='Epoch', ylabel=key); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(run / 'curves.png', dpi=160); plt.close(fig)
    scored = sorted([(v,k) for k,v in metrics['AP50'].items() if v is not None])
    best_row = max(rows, key=lambda r: float(r['mAP50']))
    highest = '、'.join(f'{k}（{v:.2%}）' for v,k in reversed(scored[-3:]))
    lowest = '、'.join(f'{k}（{v:.2%}）' for v,k in scored[:3])
    # 只有正确数据划分和相同 AP 协议的测试文件，才作为可比测试结果纳入。
    test_section = '尚未提供与本次 best.pt 对应的测试结果，不将验证分数当作测试分数。'
    test_path = run / 'test_metrics.json'
    if test_path.exists():
        test = json.loads(test_path.read_text())
        same_checkpoint = Path(test.get('checkpoint','')).resolve() == (run / 'best.pt').resolve()
        if (test.get('split') == 'test' and test.get('year') == config['year']
                and test.get('ap_protocol') == metrics['ap_protocol'] and same_checkpoint):
            score = test['mAP50']
            if score is not None:
                gap = (metrics['mAP50'] - score) * 100
                test_section = (f"最终测试 mAP@0.5 为 **{score:.4f}（{score:.2%}）**，"
                    f"评估图片 {test['images']} 张。验证减测试的差值为 {gap:.2f} 个百分点。"
                    '该差异不能单独证明过拟合，需结合数据分布与预测案例分析。\n\n'
                    '| 测试类别 | 测试 AP@0.5 |\n|---|---:|\n' + ap_table(test))
        else:
            test_section = '检测到测试文件，但其划分、年份、AP 协议或 checkpoint 路径与本实验不匹配，未纳入。'
    initialization = (f"由本项目已有检查点 `{config['init_from']}` 初始化；该检查点来自第 "
                      f"{config.get('source_epoch','未知')} 轮。这是继续训练，优化器重新初始化。"
                      if config.get('init_from') else '模型随机初始化，从零训练。')
    scales = config.get('multiscale')
    scale_text = f"训练时随机短边尺度 {sorted(set(scales + [config['image_size']]))}，验证/测试固定短边 {config['image_size']}。" if scales else f"训练及评估短边固定为 {config['image_size']}。"
    text = f'''# 机器视觉技术期末实验报告

姓名：{name}　学号：{student_id}

## 任务与方法
使用 PASCAL VOC 的 20 个目标类别完成目标检测。设计轻量残差卷积骨干，使用 GroupNorm 和 SiLU，提取步长为 8、16、32 的三个尺度特征；FPN 将通道统一为 128。检测部分复用 torchvision Faster R-CNN 的 RPN、RoIAlign 和分类回归组件，不声称完整原创检测器。{initialization}未使用外部预训练权重。

## 数据与实验设置
VOC{config['year']} 官方 train 用于训练、val 用于模型选择。训练移除 difficult 标注，评估匹配到 difficult 目标的预测不计入正负例。训练增强包括水平翻转和颜色/亮度扰动。训练图片 {config['train_images']} 张，验证图片 {config['val_images']} 张。

初始学习率 {config['lr']}，batch size {config['batch_size']}，本实验计划 {config['epochs']} 轮，已完成 {len(rows)} 轮，随机种子 {config['seed']}。模型参数 {config['parameters']:,} 个，设备 {config['device_used']}。学习率策略 {config.get('schedule','step')}，预热 {config.get('warmup_iters',0)} 次迭代。{scale_text}

## 验证集结果
以下数据来自本次日志。评估采用 IoU=0.5 的 {metrics['ap_protocol']} AP，非 COCO mAP@[.5:.95]。最佳验证 mAP@0.5 为 **{metrics['mAP50']:.4f}（{metrics['mAP50']:.2%}）**。无非困难目标的类别不计入均值。

{'注意：本次设置了 limit，仅为子集调试结果，不代表完整 VOC 实验。' if config['limit'] else ''}

![训练曲线](curves.png)

| 验证类别 | 验证 AP@0.5 |
|---|---:|
{ap_table(metrics)}

## 测试集结果
{test_section}

## 结果分析
本次训练第一轮平均损失为 {float(rows[0]['loss']):.4f}，末轮为 {float(rows[-1]['loss']):.4f}。已记录轮次中，第 {best_row['epoch']} 轮验证 mAP 最高，为 {float(best_row['mAP50']):.4f}。若从已有权重初始化，best.pt 也可能保留继续训练前的模型，不能将上述轮次误称为最终权重所属轮次。

验证集上表现较好的类别为 {highest}；较弱的类别为 {lowest}。小目标、遮挡、复杂背景和类别样本差异是待验证的可能原因，需结合预测框与真实标注逐项检查，不能仅凭 AP 表确定原因。

仅凭训练损失下降不能证明泛化性能提高，也不能单独判断过拟合。调参依据验证集；测试集仅用于最终报告，不用于挑选学习率或输入尺寸。建议保存多张成功、漏检、误检和定位偏差案例，并在此补充具体图片分析。

## 改进实验与局限
新增的继续训练、线性预热、余弦学习率和多尺度输入提供可复现的对照条件，不代表已经提高指标。先固定输入为 512 比较继续训练，再比较 640 与多尺度配置，记录额外训练成本。继续训练的总训练轮数须包含初始化检查点此前的训练，不应与仅训练 30 轮的基线声称等预算比较。

自定义骨干从零学习，数据量有限；小目标和遮挡目标仍可能具有挑战。应报告实际验证变化；若没有提升，应保留基线并说明负结果。

## 复现与参考
复现命令见项目 README。参考 torchvision Faster R-CNN、FeaturePyramidNetwork、PASCAL VOC 评估协议以及 PyTorch 学习率调度文档。应如实注明第三方组件与工具辅助。
'''
    # 生成前备份已有报告，以保留手工分析及其他未结构化内容。
    if path.exists():
        from datetime import datetime
        backup = run / ('实验报告_备份_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.md')
        backup.write_text(old)
    path.write_text(text)
    print(path)
