from myancher import multibox_target
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

def cls_eval(cls_preds, cls_labels):
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())
def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks).sum()))

def train_fn(num_epochs, model, train_iter, trainer, calc_loss, device):  
    writer=SummaryWriter("logs")  
    for epoch in range(num_epochs):
        # 训练精确度的和，训练精确度的和中的示例数
        # 绝对误差的和，绝对误差的和中的示例数
        print(f'Starting epoch {epoch + 1}/{num_epochs}...')
        model.train()
        #target的shape为[图片数，边界框数，种类]
        with tqdm.tqdm(train_iter,desc="开始训练......",position=1,leave=False) as train_iter:
            for features, label in train_iter:
                trainer.zero_grad()
                X, Y = features.to(device), label.to(device)
                # 生成多尺度的锚框，为每个锚框预测类别和偏移量
                anchors, cls_preds, bbox_preds = model(X)
                # 为每个锚框标注类别和偏移量
                bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
                # 根据类别和偏移量的预测和标注值计算损失函数
                l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                            bbox_masks)
                l.mean().backward()#对batch_size个样本求平均
                trainer.step()
                cls_err=cls_eval(cls_preds, cls_labels)/cls_labels.numel()
                bbox_mae=bbox_eval(bbox_preds, bbox_labels, bbox_masks)/bbox_labels.numel()
        print(f'Epoch {epoch + 1}, 'f'loss {l.mean():.4f}')
        cls_err=cls_eval(cls_preds, cls_labels)/cls_labels.numel()
        bbox_mae=bbox_eval(bbox_preds, bbox_labels, bbox_masks)/bbox_labels.numel()
        writer.add_scalars("loss class_err bbox_mae",{'loss_class':l.mean(),'class_err':cls_err,'bbox_mae':bbox_mae},epoch)
        torch.save(model.state_dict(),f'model_{epoch + 1}.pth')
    writer.close()