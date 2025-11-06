
from myancher import multibox_target
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

#全部样本精度和边界框绝对误差

def cls_eval(cls_preds, cls_labels):
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())
def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks).sum()))





def eval_positive_fn(model, valid_iter, device):
    model.eval()
    cls_correct_sum, bbox_mae_sum, total_positives = 0.0, 0.0, 0
    with torch.no_grad():
        for features, label in valid_iter:
            X, Y = features.to(device), label.to(device)
            # 生成多尺度的锚框，为每个锚框预测类别和偏移量
            anchors, cls_preds, bbox_preds = model(X)
            # 为每个锚框标注类别和偏移量
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            foreground_mask = (cls_labels > 0)
            num_positives_in_batch = foreground_mask.sum()
            print("num_positives_in_batch:", num_positives_in_batch.item())
            if num_positives_in_batch > 0:
                preds_on_fg = cls_preds[foreground_mask]
                labels_on_fg = cls_labels[foreground_mask]
                value = (preds_on_fg.argmax(dim=-1) == labels_on_fg).sum().item()
                print("value:", value)
                cls_correct_sum += value
            

            bbox_mae_sum += (torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum().item()
            total_positives += num_positives_in_batch.item()
            
    # 计算平均分类精度（只在正样本上）
    avg_cls_acc = cls_correct_sum / total_positives if total_positives > 0 else 0.0
    # 计算平均边界框MAE（在所有图片上）
    avg_bbox_mae = bbox_mae_sum / total_positives if total_positives > 0 else 0.0
    return avg_cls_acc, avg_bbox_mae

def train_fn(num_epochs, model, train_iter, valid_iter,trainer, calc_loss, device):    
    writer=SummaryWriter('logs/logs')  
    best_acc=0.0
    patience=10
    for epoch in range(num_epochs):
        # 训练精确度的和，训练精确度的和中的示例数
        # 绝对误差的和，绝对误差的和中的示例数
        print(f'Starting epoch {epoch + 1}/{num_epochs}...')
        model.train()
        #target的shape为[图片数，边界框数，种类]
        loss_sum=0.0
        num_batchs=0
        with tqdm.tqdm(train_iter,desc="开始训练......",position=1,leave=False) as train_iter:
            for features, label in train_iter:
                model.train()
                trainer.zero_grad()
                X, Y = features.to(device), label.to(device)
                # 生成多尺度的锚框，为每个锚框预测类别和偏移量
                #[1,num_anchors,4],[bs,num_anchors,num_classes+1],[bs,num_anchors,4]
                anchors, cls_preds, bbox_preds = model(X)
                # 为每个锚框标注类别和偏移量，类别从0开始，背景为0，前景类别从1开始
                bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
                # 根据类别和偏移量的预测和标注值计算损失函数
                l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                            bbox_masks)
                loss_of_batch=l.mean()
                loss_of_batch.backward()#对batch_size个样本求平均
                trainer.step()
                loss_sum+=loss_of_batch.item()
                num_batchs += 1
        avg_loss=loss_sum/num_batchs
        print(f'Epoch {epoch + 1}, 'f'loss {avg_loss:.4f}')
        avg_cls_acc, avg_bbox_mae = eval_positive_fn(model, valid_iter, device)
        print(f'Validation - 'f'avg_cls_acc {avg_cls_acc:.15f}')
        print(f'Validation - 'f'avg_bbox_mae {avg_bbox_mae:.15f}')
        if avg_cls_acc>=best_acc:
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with accuracy: {avg_cls_acc:.4f}')
            best_acc=avg_cls_acc
            patience=5
        else:
            patience-=1
            if patience==0:
                print("Early stopping...in epoch {}".format(epoch+1))
                break
        writer.add_scalar("train_loss", avg_loss, epoch)
        writer.add_scalars("acc_mae",{'valid_cls_acc':avg_cls_acc,'avg_bbox_mae':avg_bbox_mae},epoch)


    writer.close()
    model.load_state_dict(torch.load('best_model.pth'))