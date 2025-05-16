This repository contains the core implementation of our method described in the paper. To maintain clarity and brevity, we have included only the essential code components. The implementation uses the CIFAR-100 dataset as an example.



step 1： python Tea_resnet56_clean_CIFAR-100.py      generating clean soft label
step 2:  python Tea_wide-34-10_adv_DPGD.py           generating robust soft label
step 3:  python CAD_distillation.py                  combing these two label to train student model
