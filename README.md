<h1 align="center"> 🛡️ Frequency-based Post-train Bayesian Attack </h1>

<div align="center">
  <a href='https://arxiv.org/pdf/2407.20836'><img src='https://img.shields.io/badge/arXiv-FPBA-red'></a>  &nbsp;
  <a href="https://github.com/nxZhai/FPBA"><img src="https://img.shields.io/badge/GitHub-FPBA-9E95B7?logo=github"></a> &nbsp;
  <a href='https://huggingface.co/Oliver1515/FPBA-CNNSpot'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Model-CNNSpot(FPBA)-blue'></a> &nbsp; 
  <!-- <br> -->
  <a href='https://huggingface.co/datasets/Oliver1515/ProGAN-Eval'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Eval%20Dataset-ProGAN--Eval-blue'></a> &nbsp;
  <br>
</div>

GANs and diffusion models have raised concerns about the robustness of AI-generated image (AIGI) detectors. We propose Frequency-based Post-train Bayesian Attack (FPBA), which combines frequency-domain perturbations with a Bayesian surrogate to improve black-box transferability without retraining. Experiments show that FPBA effectively attacks diverse detectors across generators, defenses, and compression settings.

<img width="2496" height="860" alt="image" src="https://github.com/user-attachments/assets/4fe12226-248c-427e-833d-ee97f18d279e" />


## ⚙️ Setup

### 🧪 Environment

```bash
conda create -n fpba python=3.10 -y
conda activate fpba
pip install -r requirements.txt
```


### 🗂️ Dataset

Download the Synthetic LSUN / ProGAN subsets and arrange them as follows:

| Subset | Official source |
|---|---|
| `train` | [CNNSpot Train Set](https://github.com/PeterWang512/CNNDetection#training-set) |
| `eval` | [CNNSpot Val Set](https://github.com/PeterWang512/CNNDetection#validation-set) |
| `exp` | 🤗 [ProGAN-Exp](https://huggingface.co/datasets/Oliver1515/ProGAN-Exp) |


```text
<data_root>/
├── gan_train/{0_real,1_fake}/
├── gan_val/{0_real,1_fake}/
└── gan_exp/{0_real,1_fake}/
```

Use the full `gan_train` and `gan_val` splits for training. The smaller
`gan_exp` split is intended for attack and transfer smoke tests.

### 🧠 Model weights

Download the checkpoints from the following Hugging Face repositories and
place them under `./checkpoints/`:

| Model | Hugging Face |
|---|---|
| CNNSpot | 🤗 [FPBA-CNNSpot](https://huggingface.co/Oliver1515/FPBA-CNNSpot) |
| MobileNet | 🤗 [FPBA-MobileNet](https://huggingface.co/Oliver1515/FPBA-MobileNet) |
| EvalModel | 🤗 [FPBA-EvalModel](https://huggingface.co/Oliver1515/FPBA-EvalModel) |

The expected layout is:

```text
checkpoints/
├── CNNSpot_gan.pth
├── MobileNet_gan.pth
├── DenseNet_gan.pth
├── EfficientNet_gan.pth
├── ViT_gan.pth
├── Swin_gan.pth
├── Spec_gan.pth
├── DCTA_gan.pth
├── mean_gan.pt
├── var_gan.pt
└── appended_mlp/
    ├── CNNSpot/{0,1,2}_CNNSpot_PYX_AppendedModel_AT.pth
    └── MobileNet/{0,1,2}_MobileNet_PYX_AppendedModel_AT.pth
```

`Spec_gan.pth` is the published rename of `autoGAN_gan.pth`; do not use the
unvalidated `archive/spec_gan.pth` checkpoint.

> The Hugging Face repositories store files at their repository roots. After downloading, move the three CNNSpot auxiliary files into `checkpoints/appended_mlp/CNNSpot/`, move the three MobileNet auxiliary files into `checkpoints/appended_mlp/MobileNet/`, and place the detector checkpoints and DCTA statistics directly under `checkpoints/` as shown above.

## 🧪 Experiments

FPBA supports two surrogate models: `CNNSpot` and `MobileNet`.

### 🔧 Surrogate-model training

#### 🏗️ Backbone training

Train the CNNSpot or MobileNet detector backbone with early stopping:

```bash
python train.py --mode train --earlystop True --bayes False \
    --model CNNSpot --dataset gan --data_root <data_root> \
    --exp_name train_cnnspot --checkpoints_dir ./checkpoints

python train.py --mode train --earlystop True --bayes False \
    --model MobileNet --dataset gan --data_root <data_root> \
    --exp_name train_mobilenet --checkpoints_dir ./checkpoints
```

#### 🧩 FPBA auxiliary-model training

After the backbone is available, train the three appended posterior MLPs:

```bash
python train.py --mode train --earlystop False --bayes True \
    --model CNNSpot --dataset gan --data_root <data_root> \
    --exp_name bayes_cnnspot --ckpt ./checkpoints/CNNSpot_gan.pth \
    --appmodel_ckpt_root ./checkpoints/appended_mlp/CNNSpot

python train.py --mode train --earlystop False --bayes True \
    --model MobileNet --dataset gan --data_root <data_root> \
    --exp_name bayes_mobilenet --ckpt ./checkpoints/MobileNet_gan.pth \
    --appmodel_ckpt_root ./checkpoints/appended_mlp/MobileNet
```

### 🧱 White-box attack

#### ⚡ Adversarial example generation

Generate FPBA adversarial examples with either surrogate:

```bash
python attack.py --mode attack --bayes True --attack FPBA \
    --model CNNSpot --dataset gan --data_root <data_root>/gan_exp \
    --exp_name attack_cnnspot --N 5 \
    --ckpt ./checkpoints/CNNSpot_gan.pth \
    --appmodel_ckpt_root ./checkpoints/appended_mlp/CNNSpot \
    --adv_data_path ./output/cnnspot_fpba

python attack.py --mode attack --bayes True --attack FPBA \
    --model MobileNet --dataset gan --data_root <data_root>/gan_exp \
    --exp_name attack_mobilenet --N 5 \
    --ckpt ./checkpoints/MobileNet_gan.pth \
    --appmodel_ckpt_root ./checkpoints/appended_mlp/MobileNet \
    --adv_data_path ./output/mobilenet_fpba
```

> Before attacking, the script keeps only samples that the surrogate classifies correctly. Consequently, the output count may be smaller than `gan_exp`, and the reported ASR is computed on the retained samples.

### 🌐 Black-box attack

#### 🎯 Transfer to victim models

Evaluate the generated samples on DenseNet, ViT, Swin, EfficientNet, or any
other supported victim by replacing `<victim>`:

```bash
python test.py --mode tf_atk --model <victim> --dataset gan \
    --data_root ./output/cnnspot_fpba \
    --ckpt ./checkpoints/<victim>_gan.pth \
    --tf_attack FPBA --surrogate CNNSpot --exp_name transfer_cnnspot
```

Supported victims include `CNNSpot`, `DenseNet`, `EfficientNet`, `MobileNet`,
`Spec`, `DCTA`, `ViT`, and `Swin`. For `DCTA`, add
`--dcta_ckpt_dir ./checkpoints` to load `mean_gan.pt` and `var_gan.pt`.
Repeat the command with `./output/mobilenet_fpba` and
`--surrogate MobileNet` to evaluate the MobileNet surrogate.

## 📚 Citation

If you find this paper useful for your research, please use the following BibTeX entry.

```bibtex
@article{diao2026vulnerabilities,
  title={Vulnerabilities in ai-generated image detection: The challenge of adversarial attacks},
  author={Diao, Yunfeng and Zhai, Naixin and Miao, Changtao and Yu, Zitong and Wei, Xingxing and Yang, Xun and Wang, Meng},
  journal={IEEE Transactions on Multimedia},
  year={2026},
  publisher={IEEE}
}
```

## 🙏 Acknowledgement

We thank the authors of [Spectrum Simulation Attack (SSA)](https://github.com/yuyang-long/SSA) for their open-source implementation.
