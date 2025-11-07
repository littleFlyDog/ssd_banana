import torch
from model import TinySSD
from dataprocessing import load_data_bananas 
from train import train_fn 
import model
import torchvision
from predict import predict_and_display


model = TinySSD(num_classes=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#超参数设置
config={
    "batch_size": 32,
    "learning_rate": 0.05,
    "weight_decay": 5e-4,
    "num_epochs": 100,
}
train_data,valid_data= load_data_bananas(config["batch_size"])
#定义优化器
lr=config["learning_rate"]
trainer = torch.optim.SGD(model.parameters(), lr=lr,
                            weight_decay=config["weight_decay"])

#定义损失函数
cls_loss = torch.nn.CrossEntropyLoss(reduction='none')
bbox_loss = torch.nn.L1Loss(reduction='none')
def calc_loss(cls_preds, cls_labels, bbox_preds,
              bbox_labels, bbox_masks):
    batch_size, num_classes= cls_preds.shape[0], cls_preds.shape[2]
    # 计算类别损失。注意：在CrossEntropyLoss中不需要对cls_preds做softmax运算
    cls = cls_loss(cls_preds.reshape(-1,num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)#shape为[batch_size，]为不同图片的误差
    # 计算边界框损失
    bbox = bbox_loss(bbox_preds * bbox_masks,bbox_labels * bbox_masks).mean(dim=1)#shape为[batch_size，]为不同图片的误差
    return cls + bbox

#------------------训练模型----------------------
num_epochs = config["num_epochs"]
model.to(device)
train_fn(num_epochs, model, train_data, valid_data,trainer, calc_loss, device,config)




#------------------预测模型----------------------
# model.load_state_dict(torch.load('best_model_4_0.14.pth'))
# img_torch = torchvision.io.read_image('data/bananas_test/images/94.png').unsqueeze(0).float()
# model.to(device)
# predict_and_display(img_torch,model,0.9,device)