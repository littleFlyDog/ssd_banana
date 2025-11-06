import torch
from model import TinySSD
from dataprocessing import load_data_bananas 
from train import train_fn 
import model

batch_size = 32
train_data= load_data_bananas(batch_size)

model = TinySSD(num_classes=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#定义优化器
trainer = torch.optim.SGD(model.parameters(), lr=0.2,
                            weight_decay=5e-4)

#定义损失函数
cls_loss = torch.nn.CrossEntropyLoss(reduction='none')
bbox_loss = torch.nn.L1Loss(reduction='none')
def calc_loss(cls_preds, cls_labels, bbox_preds,
              bbox_labels, bbox_masks):
    batch_size, num_classes= cls_preds.shape[0], cls_preds.shape[2]
    # 计算类别损失。注意：在CrossEntropyLoss中不需要对cls_preds做softmax运算
    cls = cls_loss(cls_preds.reshape(-1,num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    # 计算边界框损失
    bbox = bbox_loss(bbox_preds * bbox_masks,bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

#------------------训练模型----------------------
num_epochs = 10
model.to(device)
train_fn(num_epochs, model, train_data, trainer, calc_loss, device)

