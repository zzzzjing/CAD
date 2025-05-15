import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18



def to_onehot(labels, num_classes):
    if labels.dim() == 1:
        return F.one_hot(labels, num_classes).float()
    elif labels.dim() == 4:
        B, dim, H, W = labels.shape
        assert dim == 1, f"Expected shape [B,1,H,W] but got {labels.shape}"
        onehot = F.one_hot(labels[:, 0].long(), num_classes=num_classes)
        return onehot.permute(0, 3, 1, 2).float()
    else:
        raise ValueError("Unsupported label dimensions")


def DiceLoss(input, target, squared_pred=False, smooth_nr=1e-5, smooth_dr=1e-5):
    intersection = torch.sum(target * input)
    if squared_pred:
        ground_o = torch.sum(target ** 2)
        pred_o = torch.sum(input ** 2)
    else:
        ground_o = torch.sum(target)
        pred_o = torch.sum(input)
    return 1.0 - (2.0 * intersection + smooth_nr) / (ground_o + pred_o + smooth_dr)


def dpgd_attack(model, images, labels, epsilon, alpha, num_iter, mean, std, num_classes):
    device = images.device
    lower = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device).view(1, 3, 1, 1)
    upper = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device).view(1, 3, 1, 1)
    eps = torch.tensor([epsilon / s for s in std], device=device).view(1, 3, 1, 1)
    alpha = torch.tensor([alpha / s for s in std], device=device).view(1, 3, 1, 1)

    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    for i in range(num_iter):
        outputs = model(adv_images)
        preds = torch.argmax(outputs, dim=1)
        correct_mask = (labels == preds)
        onehot = to_onehot(labels, num_classes)
        softmax_out = F.softmax(outputs, dim=1)

        loss_c = DiceLoss(softmax_out[correct_mask], onehot[correct_mask], squared_pred=True) if correct_mask.sum() > 0 else torch.zeros(1, device=device)
        loss_w = DiceLoss(softmax_out[~correct_mask], onehot[~correct_mask], squared_pred=True) if (~correct_mask).sum() > 0 else torch.zeros(1, device=device)

        lam = (i - 1) / (2.0 * num_iter)
        loss = (1 - lam) * loss_c + lam * loss_w

        model.zero_grad()
        loss.backward()
        adv_images = adv_images + alpha * adv_images.grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=lower, max=upper).detach_()
        adv_images.requires_grad = True

    return adv_images



def get_resnet18_cifar100():
    model = resnet18(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model



class WideBasic(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = nn.Conv2d(in_planes, out_planes, 1, stride, 0, bias=False) if not self.equalInOut else None
        self.dropRate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        shortcut = x if self.equalInOut else self.convShortcut(x)
        return out + shortcut


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super().__init__()
        layers = [block(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, dropRate) for i in range(nb_layers)]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super().__init__()
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = nn.Conv2d(3, nStages[0], 3, 1, 1, bias=False)
        self.block1 = NetworkBlock(n, nStages[0], nStages[1], WideBasic, 1, dropRate)
        self.block2 = NetworkBlock(n, nStages[1], nStages[2], WideBasic, 2, dropRate)
        self.block3 = NetworkBlock(n, nStages[2], nStages[3], WideBasic, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nStages[3], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn1(x))
        x = F.adaptive_avg_pool2d(x, 1)
        return self.fc(x.view(x.size(0), -1))



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
   
    resnet18_model = get_resnet18_cifar100().to(device)
    optimizer = optim.SGD(resnet18_model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, 101):
        resnet18_model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = resnet18_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"ResNet18 Epoch {epoch} completed.")
    torch.save(resnet18_model.state_dict(), "resnet18_cifar100_trained.pth")

    model_wr = WideResNet(depth=34, num_classes=100, widen_factor=10).to(device)
    optimizer_wr = optim.SGD(model_wr.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
    resnet18_model.eval()
    for epoch in range(1, 51):
        model_wr.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            adv_inputs = dpgd_attack(resnet18_model, inputs, targets, 8 / 255, 2 / 255, 10, mean, std, 100)
            optimizer_wr.zero_grad()
            outputs = model_wr(adv_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_wr.step()
        print(f"WideResNet Epoch {epoch} completed.")
    torch.save(model_wr.state_dict(), "wideresnet34_10_adv_trained_DPGD.pth")

    model_wr.eval()
    soft_labels = []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            adv_inputs = dpgd_attack(resnet18_model, inputs, targets.to(device), 8 / 255, 2 / 255, 10, mean, std, 100)
            outputs = model_wr(adv_inputs)
            soft = F.softmax(outputs, dim=1)
            soft_labels.append(soft.cpu())
    torch.save(torch.cat(soft_labels, dim=0), "adv_soft_labels_DPGD.pth")
    print("Soft labels saved.")
