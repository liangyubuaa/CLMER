import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
def make_confusion_matrix(true, pred, class_name, dataset, index, mode):
    # 计算规范化后的混淆矩阵，normalize='pred'
    color = 'Blues'
    if 'deap' in dataset:
        color = 'Reds'
    elif 'amigos' in dataset:
        color = 'Purples'
    elif 'private' in dataset:
        color = 'Oranges'
    else:
        color = 'Blues'
    
    true = true.cpu().numpy().squeeze()
    pred = np.argmax(pred.cpu().numpy().squeeze(),axis=1)
    cm_normalized = confusion_matrix(true, pred, normalize='pred')
    # 绘制热图
    plt.figure(figsize=(12, 10))

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=color, xticklabels=class_name, yticklabels=class_name)

    # 添加标题和标签
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Normalized Confusion Matrix (Pred)')

    # 保存图像
    plt.savefig(f'confusion_matrix/{dataset}{index}{mode}.png', dpi=300)  # dpi可以根据需要调整
    plt.close()  # 关闭图形以释放内存