import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.backends import cudnn
from torch.cuda.amp import autocast  # 4090混合精度加速

# 导入你的自定义模块
from fer import Fer2013  # 你的FER2013 Dataset类
from models.resnet import ResNet18  # 训练好的ResNet18（带Dropout）
# 如果要测试VGG19，取消注释
from models.vgg import VGG19

# ===================== 核心配置（按需修改） =====================
# 1. 训练好的模型权重路径（替换为你的resnet_original.pth）
# MODEL_WEIGHT_PATH = "./weights/resnet_original.pth"
MODEL_WEIGHT_PATH = "./weights/emotion_model.pth"
# 2. FER2013数据路径（固定为你提供的./data/fer2013.h5）
FER_H5_PATH = "./data/fer2013.h5"
# 3. 测试集类型："test"（泛化能力）或 "train"（拟合能力）
TEST_SPLIT = "test"
# 4. 设备配置（优先5090 GPU）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 5. 超参数（和训练时一致）
BATCH_SIZE = 128
NUM_CLASSES = 7
# 6. 表情标签映射（完全匹配你提供的EMOTION_MAP）
EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}
# 反向映射（方便打印）
IDX_TO_EMOTION = {v: k for k, v in EMOTION_MAP.items()}

# ===================== 数据变换（必须和训练时一致！） =====================
# 测试集变换（如果训练时用了10-Crop，用这个版本）
# test_transform = transforms.Compose([
#     transforms.TenCrop(44),  # 和训练时的CenterCrop(44)匹配
#     transforms.Lambda(lambda crops: torch.stack([
#         transforms.Compose([
#             transforms.ToTensor(),
#             # 训练时如果加了标准化，这里必须加！
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])(crop) for crop in crops
#     ])),
# ])

# 如果你训练时没用到10-Crop，用这个基础版本：
test_transform = transforms.Compose([
    transforms.CenterCrop(44),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===================== 加载FER2013数据集 =====================
def load_fer_dataset():
    """加载FER2013的train/test集"""
    # 验证h5文件是否存在
    if not os.path.exists(FER_H5_PATH):
        print(f"❌ 找不到FER2013数据文件：{FER_H5_PATH}")
        exit(1)
    
    test_dataset = Fer2013(
        split=TEST_SPLIT,
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,  # 4090适配：CPU核心数的1/2
        pin_memory=True,  # GPU数据传输加速
        persistent_workers=True  # 保持数据加载进程
    )
    print(f"✅ FER2013 {TEST_SPLIT}集加载完成，共{len(test_dataset)}个样本")
    print(f"   - 训练集：28709样本 | 测试集：7178样本（和你的h5结构一致）")
    return test_loader

# ===================== 加载训练好的模型 =====================
def load_model():
    """加载ResNet18/VGG19模型，加载训练权重"""
    # 初始化模型（和训练时的结构完全一致！）
    # model = ResNet18(num_classes=NUM_CLASSES).to(DEVICE)
    # 如果测试VGG19，替换为：
    model = VGG19(num_classes=NUM_CLASSES).to(DEVICE)
    
    # 加载权重（适配GPU/CPU）
    try:
        weight_dict = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE)
        model.load_state_dict(weight_dict)
        print(f"✅ 模型权重加载成功：{MODEL_WEIGHT_PATH}")
    except Exception as e:
        print(f"❌ 权重加载失败：{e}")
        print("请检查：1.模型路径是否正确 2.模型结构是否和训练时一致（比如Dropout）")
        exit(1)
    
    # 切换到预测模式（关闭Dropout/BatchNorm训练行为）
    model.eval()
    return model

# ===================== 执行预测并计算指标 =====================
def predict_on_fer(model, test_loader):
    """在FER2013上执行预测，计算整体/每类准确率"""
    criterion = nn.CrossEntropyLoss()  # 计算损失（可选）
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 统计每类的正确数/总数（匹配EMOTION_MAP）
    class_correct = np.zeros(NUM_CLASSES, dtype=int)
    class_total = np.zeros(NUM_CLASSES, dtype=int)

    print(f"\n开始在FER2013 {TEST_SPLIT}集上预测...")
    with torch.no_grad():  # 关闭梯度，4090提速+省显存
        for batch_idx, (images, labels) in enumerate(test_loader):
            labels = labels.to(DEVICE)
            
            # 处理10-Crop数据（如果用了10-Crop）
            if len(images.size()) == 5:  # (batch, 10, 3, 44, 44)
                bs, ncrops, c, h, w = images.size()
                images = images.view(-1, c, h, w).to(DEVICE)  # (batch*10, 3, 44, 44)
                
                # 混合精度前向传播（5090加速）
                with autocast():
                    outputs = model(images)
                    # 10-Crop取平均
                    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
                    loss = criterion(outputs_avg, labels)
                
                # 统计预测结果
                _, predicted = torch.max(outputs_avg, 1)
                batch_size_current = bs
            else:  # 基础版本（无10-Crop）
                images = images.to(DEVICE)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs, 1)
                batch_size_current = images.size(0)
            
            # 统计整体损失和准确率
            total_loss += loss.item() * batch_size_current
            total += batch_size_current
            correct += predicted.eq(labels).sum().item()
            
            # 统计每类的正确数/总数
            for i in range(batch_size_current):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
            
            # 打印批次进度
            if (batch_idx + 1) % 20 == 0:
                batch_acc = 100. * predicted.eq(labels).sum().item() / batch_size_current
                print(f"批次 [{batch_idx+1}/{len(test_loader)}] | 批次损失：{loss.item():.4f} | 批次准确率：{batch_acc:.2f}%")

    # 计算整体指标
    avg_loss = total_loss / total
    overall_acc = 100. * correct / total

    # 打印结果
    print("\n" + "="*70)
    print(f"📊 FER2013 {TEST_SPLIT}集预测结果（ResNet18）")
    print(f"="*70)
    print(f"整体平均损失：{avg_loss:.4f}")
    print(f"整体准确率：{overall_acc:.2f}% ({correct}/{total})")
    print(f"="*70)

    # 打印每类表情准确率（核心分析模型表现）
    print("\n🎯 每类表情准确率：")
    print("-"*60)
    print(f"{'表情类别':<10} {'索引':<5} {'准确率':<10} {'正确数/总数'}")
    print("-"*60)
    for idx in range(NUM_CLASSES):
        if class_total[idx] > 0:
            class_acc = 100. * class_correct[idx] / class_total[idx]
            print(f"{EMOTION_MAP[idx]:<10} {idx:<5} {class_acc:.2f}%       {class_correct[idx]}/{class_total[idx]}")
        else:
            print(f"{EMOTION_MAP[idx]:<10} {idx:<5} 无样本         0/0")
    print("-"*60)

    # 模型表现分析
    print("\n📈 模型表现分析：")
    if TEST_SPLIT == "train":
        if overall_acc > 90:
            print(f"- 拟合能力：优秀（训练集准确率>90%，模型学到了特征）")
        elif overall_acc > 80:
            print(f"- 拟合能力：中等（训练集准确率80%-90%，拟合不充分）")
        else:
            print(f"- 拟合能力：较差（训练集准确率<80%，模型未学到核心特征）")
    else:  # test集
        if overall_acc > 70:
            print(f"- 泛化能力：优秀（测试集准确率>70%，跨样本泛化好）")
        elif overall_acc > 65:
            print(f"- 泛化能力：中等（测试集准确率65%-70%，轻微过拟合）")
        else:
            print(f"- 泛化能力：较差（测试集准确率<65%，严重过拟合/欠拟合）")

# ===================== 主函数 =====================
if __name__ == "__main__":
    # 5090 GPU加速配置
    cudnn.benchmark = True
    print(f"🔧 使用设备：{DEVICE}")
    if torch.cuda.is_available():
        print(f"🔧 GPU型号：{torch.cuda.get_device_name(0)}")
        print(f"🔧 GPU显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 1. 加载模型
    model = load_model()
    
    # 2. 加载FER2013数据集
    test_loader = load_fer_dataset()
    
    # 3. 执行预测
    predict_on_fer(model, test_loader)