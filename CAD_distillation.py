import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


# 辅助函数定义
def to_onehot(labels, num_classes):
    # 支持标签形状为 [B]、[B,1] 或标量
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)
    if labels.dim() == 2 and labels.size(1) == 1:
        labels = labels.squeeze(1)
    if labels.dim() == 1:
        onehot = F.one_hot(labels.long(), num_classes=num_classes).float()
    elif labels.dim() == 4:
        B, dim, H, W = labels.shape
        assert dim == 1, f"Invalid 'labels' shape. Expected [B,1,H,W] but got {labels.shape}"
        onehot = F.one_hot(labels[:, 0].long(), num_classes=num_classes)
        onehot = onehot.permute(0, 3, 1, 2).float()
    else:
        raise ValueError("Unsupported label dimensions")
    return onehot

def DiceLoss(input, target, squared_pred=False, smooth_nr=1e-5, smooth_dr=1e-5):
    intersection = torch.sum(target * input)
    if squared_pred:
        ground_o = torch.sum(target ** 2)
        pred_o = torch.sum(input ** 2)
    else:
        ground_o = torch.sum(target)
        pred_o = torch.sum(input)
    denominator = ground_o + pred_o
    dice_loss = 1.0 - (2.0 * intersection + smooth_nr) / (denominator + smooth_dr)
    return dice_loss

def mpgd_attack(model, images, labels, epsilon, alpha, num_iter, mean, std, num_classes):
    device = images.device
    # 计算归一化空间下图像的合法取值范围
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)],
                               device=device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)],
                               device=device).view(1, 3, 1, 1)
    # epsilon 与 alpha 转换到归一化空间（每个通道除以对应的 std）
    epsilon_tensor = torch.tensor([epsilon / s for s in std],
                                  device=device).view(1, 3, 1, 1)
    alpha_tensor = torch.tensor([alpha / s for s in std],
                                device=device).view(1, 3, 1, 1)
    # 初始化对抗样本
    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    for i in range(num_iter):
        outputs = model(adv_images)  # 输出 shape 为 [B, num_classes]
        pred_labels = torch.argmax(outputs, dim=1)  # [B]
        correct_mask = (labels == pred_labels)  # 布尔型张量
        onehot = to_onehot(labels, num_classes)  # shape [B, num_classes]
        adv_pred_softmax = F.softmax(outputs, dim=1)  # shape [B, num_classes]

        # 分别计算正确预测样本和错误预测样本的 DiceLoss
        if correct_mask.sum() > 0:
            loss_correct = DiceLoss(adv_pred_softmax[correct_mask],
                                    onehot[correct_mask],
                                    squared_pred=True)
        else:
            loss_correct = 0.0 * adv_images.sum()
        if (~correct_mask).sum() > 0:
            loss_wrong = DiceLoss(adv_pred_softmax[~correct_mask],
                                  onehot[~correct_mask],
                                  squared_pred=True)
        else:
            loss_wrong = 0.0 * adv_images.sum()

        # 动态权重：随着迭代步数逐渐增加，lambda 由 0 逐步上升
        lam = (i - 1) / (2.0 * num_iter)
        loss = (1 - lam) * loss_correct + lam * loss_wrong

        model.zero_grad()
        loss.backward()

        # 根据梯度符号更新对抗样本
        adv_images = adv_images + alpha_tensor * adv_images.grad.sign()

        # 对扰动进行投影，确保每个像素扰动不超过 epsilon
        perturbation = torch.clamp(adv_images - images, min=-epsilon_tensor, max=epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images


# Dataset：知识蒸馏数据集

class DistillationDataset(Dataset):
    def __init__(self, base_dataset, teacher_soft_labels):

        self.base_dataset = base_dataset
        self.teacher_soft_labels = teacher_soft_labels

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        teacher_target = self.teacher_soft_labels[index]
        return img, target, teacher_target


# 主函数
def main():
    # 超参数设置
    num_classes = 100
    batch_size = 128
    num_epochs = 100
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    # PGD 攻击参数
    epsilon = 8.0/255.0
    alpha = 2.0/255.0
    num_iter = 10

    # CIFAR-100 归一化参数
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)

    # 知识蒸馏相关参数
    distill_alpha = 0.7    # KL 损失权重
    temperature = 1.0      # 温度参数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    # 加载 CIFAR100 数据集
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)


    # 加载教师模型的 soft label 文件，并取平均
    teacher_soft_path1 = "advtrain_soft_labels_DPGD.pth"
    teacher_soft_path2 = "train_soft_labels.pth"

    if os.path.exists(teacher_soft_path1) and os.path.exists(teacher_soft_path2):
        teacher_soft1 = torch.load(teacher_soft_path1)
        teacher_soft2 = torch.load(teacher_soft_path2)
        # 取平均得到融合后的教师 soft label，确保数据在 CPU 上
        teacher_soft_labels = (teacher_soft1 +  teacher_soft2)/2
        print("成功加载并融合两个教师的 soft label 文件")
    else:
        raise FileNotFoundError("未找到教师 soft label 文件，请检查 teacher_soft_path1 与 teacher_soft_path2")

    # 构建知识蒸馏专用的数据集
    distill_dataset = DistillationDataset(trainset, teacher_soft_labels)
    distill_loader = DataLoader(distill_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


    # 学生模型,ResNet18
    student_model = torchvision.models.resnet18(weights=None)
    student_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    student_model.maxpool = nn.Identity()
    student_model.fc = nn.Linear(student_model.fc.in_features, num_classes)
    student_model = student_model.to(device)

    optimizer = optim.SGD(student_model.parameters(), lr=learning_rate,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    # KLDivLoss 用于知识蒸馏
    kl_loss = nn.KLDivLoss(reduction='batchmean')


    # 对学生模型进行对抗知识蒸馏训练
    print("开始对学生模型进行对抗知识蒸馏训练 ...")
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        for inputs, targets, teacher_soft in tqdm(distill_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            teacher_soft = teacher_soft.to(device)  # 形状 [B, num_classes]

            # 使用学生模型自身生成对抗样本
            adv_inputs = mpgd_attack(student_model, inputs, targets, epsilon, alpha, num_iter, cifar100_mean, cifar100_std, num_classes)
            optimizer.zero_grad()
            outputs = student_model(adv_inputs)

            # 知识蒸馏损失
            log_probs = F.log_softmax(outputs / temperature, dim=1)
            teacher_probs = F.softmax(teacher_soft / temperature, dim=1)
            loss_KL = kl_loss(log_probs, teacher_probs) * (temperature ** 2)

            # 真实标签的交叉熵损失
            loss_CE = F.cross_entropy(outputs, targets)

            # 总损失：两者加权融合
            loss = distill_alpha * loss_KL + (1.0 - distill_alpha) * loss_CE
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        avg_loss = running_loss / len(distill_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] 学生模型蒸馏训练 Loss: {avg_loss:.4f}")

        # 每个 epoch 后评估学生模型在测试集上的准确率
        student_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = student_model(inputs)
                predicted = outputs.argmax(dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] 测试集准确率: {acc:.2f}%")

    # 保存训练好的学生模型
    torch.save(student_model.state_dict(), "CAD_student_resnet18_distilled.pth")
    print("学生模型训练完成，模型参数已保存。")

if __name__ == '__main__':
    main()
