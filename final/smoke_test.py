"""不下载数据，验证模型前向/反向、所有参数梯度及推理输出。"""
import torch
from model import build_model

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    model=build_model(64)
    model.train()
    images=[torch.rand(3,64,64),torch.rand(3,64,64)]
    targets=[{'boxes':torch.tensor([[8.,8.,40.,48.]]),'labels':torch.tensor([1])} for _ in images]
    losses=model(images,targets);loss=sum(losses.values())
    assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    model.eval()
    with torch.no_grad(): predictions=model(images)
    assert len(predictions)==2
    for p in predictions:
        assert p['boxes'].shape[1]==4
        assert (p['boxes'][:,2:]>=p['boxes'][:,:2]).all()
    print('模型前向、反向和推理检查通过；loss =',loss.item())
if __name__=='__main__': main()
