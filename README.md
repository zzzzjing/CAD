**Code for NeurIPS 2025 submission "Collaborative Adversarial Distillation for Robust Defense Against Adversarial Attacks".**
---

This repository contains the core implementation of our method described in the paper. To maintain clarity, we have included only the essential code components. The implementation uses the CIFAR-100 dataset as an example.

---

**Requirements**：
<br>
Python 3.8+ <br>
PyTorch 1.10+<br>
torchvision<br>
numpy<br>
tqdm<br>
CUDA<br>
AutoAttack<br>

---

**Run**:
<br>step 1：python Tea_clean_resnet56_CIFAR-100.py      
<br>step 2: python Tea_wide-34-10_adv_DPGD.py           
<br>step 3: python CAD_distillation.py                  
<br>step 4: python eva_resnet18.py

---
