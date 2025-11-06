import torch
from myancher import multibox_detection, show_bboxes
from torch.nn import functional as F
import matplotlib.pyplot as plt



# def predict(img_torch,model,threshold,device):
#     model.eval()
#     anchors, cls_preds, bbox_preds = model(img_torch.to(device))
#     cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
#     output = multibox_detection(cls_probs, bbox_preds, anchors)
#     idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
#     results= output[0, idx]
#     img=img_torch.squeeze(0).permute(1, 2, 0).long()
#     fig,ax=plt.subplots()
#     ax.imshow(img.numpy().astype("uint8"))
#     for row in results:
#         score = float(row[1])
#         if score < threshold:
#             continue
#         h, w = img.shape[0:2]
#         bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
#         show_bboxes(ax, bbox, '%.2f' % score, 'w')
#     plt.show()



def predict_and_display(img_torch, model, threshold,device):
    """对单张图片进行预测，并将高于阈值的结果可视化显示"""
    
    # 1. 设置模型为评估模式并进行推理
    model.eval()
    with torch.no_grad():
        anchors, cls_preds, bbox_preds = model(img_torch.to(device))
        
    # 2. 对模型原始输出进行后处理
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    
    # 3. 筛选出所有有效的检测结果（前景物体）
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]#不是背景的话，就将索引加入
    results = output[0, idx]#[num_selected_anchor,6]
    
    # # 4. 准备图像用于显示 (从 CHW 转换为 HWC)
    # #    去掉不必要的 .long()
    img_display = img_torch.squeeze(0).permute(1, 2, 0)
    h, w = img_display.shape[0:2]
    
    # 5. 创建画布并显示背景图
    fig, ax = plt.subplots()
    ax.imshow(img_display.numpy().astype('uint8')) # 
    ax.axis('off')
    

    for row in results:
        score = float(row[1])
        if score >= threshold:
            # 反归一化坐标
            bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
            show_bboxes(ax, bbox, '%.2f' % score, 'w')
                
    plt.show()