import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18



# 辅助函数：to_onehot 和 DiceLoss
def to_onehot(labels, num_classes):

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



# 定义 mpgd_attack：基于多步 PGD 并结合 DiceLoss 的对抗攻击方法
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
        pred_labels = torch.argmax(outputs, dim=1)
        # 得到正确预测样本的 mask
        correct_mask = (labels == pred_labels)
        onehot = to_onehot(labels, num_classes)
        adv_pred_softmax = F.softmax(outputs, dim=1)

        # 分别计算正确预测与错误预测样本的 DiceLoss
        if correct_mask.sum() > 0:
            loss_correct = DiceLoss(adv_pred_softmax[correct_mask],
                                    onehot[correct_mask],
                                    squared_pred=True)
        else:
            loss_correct = torch.zeros(1, device=device, requires_grad=True)

        if (~correct_mask).sum() > 0:
            loss_wrong = DiceLoss(adv_pred_softmax[~correct_mask],
                                  onehot[~correct_mask],
                                  squared_pred=True)
        else:
            loss_wrong = torch.zeros(1, device=device, requires_grad=True)

        # 动态权重：随着迭代逐步增加，lambda 从 0 开始逐步上升
        lam = (i - 1) / (2.0 * num_iter)
        loss = (1 - lam) * loss_correct + lam * loss_wrong

        model.zero_grad()
        loss.backward()
        # 根据梯度符号更新对抗样本
        adv_images = adv_images + alpha_tensor * adv_images.grad.sign()
        # 投影：确保扰动幅度不超过 epsilon
        perturbation = torch.clamp(adv_images - images, min=-epsilon_tensor, max=epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images



# 定义适用于 CIFAR-100 的 ResNet18

def get_resnet18_cifar100():
    model = resnet18(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model



# WideResNet-34-10 相关定义
class WideBasic(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
                             if not self.equalInOut else None)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return self.convShortcut(x) + out
        else:
            return x + out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes if i == 0 else out_planes, out_planes,
                                stride if i == 0 else 1, dropRate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        assert ((depth - 4) % 6 == 0), 'depth 应为 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nStages[0], nStages[1], WideBasic, stride=1, dropRate=dropRate)
        self.block2 = NetworkBlock(n, nStages[1], nStages[2], WideBasic, stride=2, dropRate=dropRate)
        self.block3 = NetworkBlock(n, nStages[2], nStages[3], WideBasic, stride=2, dropRate=dropRate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nStages[3], num_classes)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# 数据加载与预处理
mean = [0.5071, 0.4867, 0.4409]
std = [0.2675, 0.2565, 0.2761]

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 训练 ResNet18 模型
print("开始训练 ResNet18 ...")
resnet18_model = get_resnet18_cifar100().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18_model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
epochs_resnet18 = 100

for epoch in range(1, epochs_resnet18 + 1):
    resnet18_model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = resnet18_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(
                f"ResNet18 Epoch [{epoch}/{epochs_resnet18}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
    print(f"Epoch {epoch} 平均 Loss: {running_loss / len(train_loader):.4f}")

torch.save(resnet18_model.state_dict(), "resnet18_cifar100_trained.pth")
print("ResNet18 模型参数已保存至 resnet18_cifar100_trained.pth")


# 对抗训练 WideresNet-34-10
model_wideresnet = WideResNet(depth=34, num_classes=100, widen_factor=10, dropRate=0.0).to(device)
optimizer_adv = optim.SGD(model_wideresnet.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
criterion_adv = nn.CrossEntropyLoss()
epochs_wideresnet = 50

# 使用训练好的 ResNet18 生成对抗样本，设置为 eval 模式，并固定参数
resnet18_model.eval()

print("开始对抗训练 WideresNet-34-10 ...")
for epoch in range(1, epochs_wideresnet + 1):
    model_wideresnet.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # 使用 mpgd_attack 方法生成对抗样本
        adv_inputs = mpgd_attack(resnet18_model, inputs, targets,
                                 epsilon=8 / 255, alpha=2 / 255, num_iter=10,
                                 mean=mean, std=std, num_classes=100)
        optimizer_adv.zero_grad()
        outputs = model_wideresnet(adv_inputs)
        loss = criterion_adv(outputs, targets)
        loss.backward()
        optimizer_adv.step()
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(
                f"WideresNet Epoch [{epoch}/{epochs_wideresnet}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
    print(f"Epoch {epoch} 平均 Loss: {running_loss / len(train_loader):.4f}")

torch.save(model_wideresnet.state_dict(), "wideresnet34_10_adv_trained_DPGD.pth")
print("WideresNet-34-10 模型参数已保存至 wideresnet34_10_adv_trained.pth")


# 使用训练后的 WideresNet-34-10 模型生成对抗样本 soft label 并保存
model_wideresnet.eval()
soft_labels_list = []

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_inputs = mpgd_attack(resnet18_model, inputs, targets,
                                 epsilon=8 / 255, alpha=2 / 255, num_iter=10,
                                 mean=mean, std=std, num_classes=100)
        outputs = model_wideresnet(adv_inputs)
        soft = F.softmax(outputs, dim=1)
        soft_labels_list.append(soft.cpu())

soft_labels_all = torch.cat(soft_labels_list, dim=0)
torch.save(soft_labels_all, "adv_soft_labels_DPGD.pth")
print("对抗样本的 soft label 已保存至 adv_soft_labels.pth")
