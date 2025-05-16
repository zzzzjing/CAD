import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_student_model(num_classes=100, device=None):
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if device is not None:
        model = model.to(device)
    return model

def fgsm_attack(model, images, labels, epsilon, mean, std):
    device = images.device
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    epsilon_tensor = torch.tensor([epsilon / s for s in std], device=device).view(1,3,1,1)

    images = images.clone().detach()
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    adv_images = images + epsilon_tensor * images.grad.sign()
    adv_images = torch.clamp(adv_images, lower_bound, upper_bound)
    return adv_images.detach()

def pgd_attack(model, images, labels, epsilon, alpha, num_iter, mean, std):
    device = images.device
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    epsilon_tensor = torch.tensor([epsilon / s for s in std], device=device).view(1,3,1,1)
    alpha_tensor = torch.tensor([alpha / s for s in std], device=device).view(1,3,1,1)

    adv_images = images.clone().detach()
    init_noise = torch.empty_like(adv_images).uniform_(-1, 1)
    init_noise = init_noise * epsilon_tensor
    adv_images = adv_images + init_noise
    adv_images = torch.clamp(adv_images, lower_bound, upper_bound)
    adv_images.requires_grad = True

    for _ in range(num_iter):
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = adv_images + alpha_tensor * adv_images.grad.sign()
        perturbation = torch.clamp(adv_images - images, -epsilon_tensor, epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images.detach()

def tpgd_attack(model, images, labels, epsilon, alpha, num_iter, mean, std):
    device = images.device
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    epsilon_tensor = torch.tensor([epsilon / s for s in std], device=device).view(1,3,1,1)
    alpha_tensor = torch.tensor([alpha / s for s in std], device=device).view(1,3,1,1)

    with torch.no_grad():
        outputs = model(images)
        target_labels = outputs.argmin(dim=1)

    adv_images = images.clone().detach()
    init_noise = torch.empty_like(adv_images).uniform_(-1, 1)
    init_noise = init_noise * epsilon_tensor
    adv_images = adv_images + init_noise
    adv_images = torch.clamp(adv_images, lower_bound, upper_bound)
    adv_images.requires_grad = True

    for _ in range(num_iter):
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, target_labels)
        model.zero_grad()
        loss.backward()
        adv_images = adv_images - alpha_tensor * adv_images.grad.sign()
        perturbation = torch.clamp(adv_images - images, -epsilon_tensor, epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images.detach()

def i_fgsm_attack(model, images, labels, epsilon, alpha, num_iter, mean, std):
    device = images.device
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    epsilon_tensor = torch.tensor([epsilon / s for s in std], device=device).view(1,3,1,1)
    alpha_tensor = torch.tensor([alpha / s for s in std], device=device).view(1,3,1,1)

    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    for _ in range(num_iter):
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = adv_images + alpha_tensor * adv_images.grad.sign()
        perturbation = torch.clamp(adv_images - images, -epsilon_tensor, epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images.detach()

class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.mean = torch.tensor(mean).view(1,3,1,1)
        self.std = torch.tensor(std).view(1,3,1,1)

    def forward(self, x):
        return self.model((x - self.mean.to(x.device)) / self.std.to(x.device))

def evaluate(model, test_loader, attack_fn, **attack_kwargs):
    correct = 0
    total = 0
    model.eval()
    for inputs, labels in tqdm(test_loader, desc=attack_fn.__name__):
        inputs, labels = inputs.to(device), labels.to(device)
        adv_inputs = attack_fn(model, inputs, labels, **attack_kwargs)
        with torch.no_grad():
            outputs = model(adv_inputs)
            pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)

    epsilon = 8.0 / 255.0
    alpha = 2.0 / 255.0
    num_iter = 10

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    student_model = get_student_model(num_classes=100, device=device)
    ckpt_path = "CAD_distilled.pth"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found.")
        exit(0)
    student_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    student_model.eval()
    print("Model loaded successfully. Starting adversarial evaluation...")

    acc_fgsm = evaluate(student_model, test_loader, fgsm_attack,
                        epsilon=epsilon, mean=cifar100_mean, std=cifar100_std)
    print(f"[FGSM] Accuracy: {acc_fgsm:.2f}%")

    acc_pgd = evaluate(student_model, test_loader, pgd_attack,
                       epsilon=epsilon, alpha=alpha, num_iter=num_iter,
                       mean=cifar100_mean, std=cifar100_std)
    print(f"[PGD] Accuracy: {acc_pgd:.2f}%")

    acc_tpgd = evaluate(student_model, test_loader, tpgd_attack,
                        epsilon=epsilon, alpha=alpha, num_iter=num_iter,
                        mean=cifar100_mean, std=cifar100_std)
    print(f"[TPGD] Accuracy: {acc_tpgd:.2f}%")

    acc_ifgsm = evaluate(student_model, test_loader, i_fgsm_attack,
                         epsilon=epsilon, alpha=alpha, num_iter=num_iter,
                         mean=cifar100_mean, std=cifar100_std)
    print(f"[I-FGSM] Accuracy: {acc_ifgsm:.2f}%")

    try:
        from autoattack import AutoAttack
        print("[AutoAttack] Running evaluation...")
        all_data = []
        all_labels = []
        for imgs, lbls in test_loader:
            all_data.append(imgs)
            all_labels.append(lbls)
        all_data = torch.cat(all_data, dim=0).to(device)
        all_labels = torch.cat(all_labels, dim=0).to(device)

        mean_tensor = torch.tensor(cifar100_mean).view(1,3,1,1).to(device)
        std_tensor = torch.tensor(cifar100_std).view(1,3,1,1).to(device)
        data_unnorm = all_data * std_tensor + mean_tensor

        wrapped_model = NormalizedModel(student_model, cifar100_mean, cifar100_std).to(device)
        wrapped_model.eval()

        adversary = AutoAttack(wrapped_model, norm='Linf', eps=epsilon, version='standard')
        adv_data = adversary.run_standard_evaluation(data_unnorm, all_labels, bs=128)

        adv_data_norm = (adv_data - mean_tensor) / std_tensor
        with torch.no_grad():
            outputs = student_model(adv_data_norm)
            preds = outputs.argmax(dim=1)
        correct = (preds == all_labels).sum().item()
        acc_auto = 100.0 * correct / len(all_labels)
        print(f"[AutoAttack] Accuracy: {acc_auto:.2f}%")
    except ImportError:
        print("AutoAttack not installed. Skipping AutoAttack evaluation.")
    except Exception as e:
        print("AutoAttack evaluation failed:", e)
