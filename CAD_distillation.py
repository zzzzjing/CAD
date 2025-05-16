import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def to_onehot(labels, num_classes):
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

def dpgd_attack(model, images, labels, epsilon, alpha, num_iter, mean, std, num_classes):
    device = images.device
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device).view(1, 3, 1, 1)
    epsilon_tensor = torch.tensor([epsilon / s for s in std], device=device).view(1, 3, 1, 1)
    alpha_tensor = torch.tensor([alpha / s for s in std], device=device).view(1, 3, 1, 1)
    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    for i in range(num_iter):
        outputs = model(adv_images)
        pred_labels = torch.argmax(outputs, dim=1)
        correct_mask = (labels == pred_labels)
        onehot = to_onehot(labels, num_classes)
        adv_pred_softmax = F.softmax(outputs, dim=1)

        if correct_mask.sum() > 0:
            loss_correct = DiceLoss(adv_pred_softmax[correct_mask], onehot[correct_mask], squared_pred=True)
        else:
            loss_correct = 0.0 * adv_images.sum()
        if (~correct_mask).sum() > 0:
            loss_wrong = DiceLoss(adv_pred_softmax[~correct_mask], onehot[~correct_mask], squared_pred=True)
        else:
            loss_wrong = 0.0 * adv_images.sum()

        lam = (i - 1) / (2.0 * num_iter)
        loss = (1 - lam) * loss_correct + lam * loss_wrong

        model.zero_grad()
        loss.backward()
        adv_images = adv_images + alpha_tensor * adv_images.grad.sign()
        perturbation = torch.clamp(adv_images - images, min=-epsilon_tensor, max=epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images

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

def main():
    num_classes = 100
    batch_size = 128
    num_epochs = 100
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    epsilon = 8.0/255.0
    alpha = 2.0/255.0
    num_iter = 10
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    distill_alpha = 0.7
    temperature = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    teacher_soft_path1 = "advtrain_soft_labels_DPGD.pth"
    teacher_soft_path2 = "train_soft_labels.pth"

    if os.path.exists(teacher_soft_path1) and os.path.exists(teacher_soft_path2):
        teacher_soft1 = torch.load(teacher_soft_path1)
        teacher_soft2 = torch.load(teacher_soft_path2)
        teacher_soft_labels = (teacher_soft1 +  teacher_soft2) / 2
        print("Successfully loaded and fused teacher soft labels.")
    else:
        raise FileNotFoundError("Teacher soft label files not found.")

    distill_dataset = DistillationDataset(trainset, teacher_soft_labels)
    distill_loader = DataLoader(distill_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    student_model = torchvision.models.resnet18(weights=None)
    student_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    student_model.maxpool = nn.Identity()
    student_model.fc = nn.Linear(student_model.fc.in_features, num_classes)
    student_model = student_model.to(device)

    optimizer = optim.SGD(student_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    print("Starting adversarial distillation training for the student model ...")
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        for inputs, targets, teacher_soft in tqdm(distill_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            teacher_soft = teacher_soft.to(device)

            adv_inputs = dpgd_attack(student_model, inputs, targets, epsilon, alpha, num_iter, cifar100_mean, cifar100_std, num_classes)
            optimizer.zero_grad()
            outputs = student_model(adv_inputs)

            log_probs = F.log_softmax(outputs / temperature, dim=1)
            teacher_probs = F.softmax(teacher_soft / temperature, dim=1)
            loss_KL = kl_loss(log_probs, teacher_probs) * (temperature ** 2)
            loss_CE = F.cross_entropy(outputs, targets)
            loss = distill_alpha * loss_KL + (1.0 - distill_alpha) * loss_CE
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        avg_loss = running_loss / len(distill_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] Distillation training loss: {avg_loss:.4f}")

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
        print(f"Epoch [{epoch+1}/{num_epochs}] Test accuracy: {acc:.2f}%")

    torch.save(student_model.state_dict(), "CAD_distilled.pth")
    print("Student model training completed. Model saved.")

if __name__ == '__main__':
    main()
