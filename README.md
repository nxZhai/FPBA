<h1 align="center"> Frequency-based Post-train Bayesian Attack </h1>

<div align="center">
  <a href='https://arxiv.org/pdf/2407.20836'><img src='https://img.shields.io/badge/arXiv-FPBA-red'></a>  &nbsp;
  <a href="https://github.com/onotoa/fpba"><img src="https://img.shields.io/badge/GitHub-FPBA-9E95B7?logo=github"></a> &nbsp; 
  <a href='https://huggingface.co/Oliver1515/FPBA-CNNSpot'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Model-CNNSpot(FPBA)-blue'></a> &nbsp; 
  <!-- <br> -->
  <a href='https://huggingface.co/datasets/Oliver1515/ProGAN-Eval'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Eval%20Dataset-ProGAN--Eval-blue'></a> &nbsp;
  <br>
</div>

Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. To this end, we propose a new method to attack AIGI detectors. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous AIGI detectors, e.g., transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as Frequency-based Post-train Bayesian Attack, or FPBA. Through FPBA, we demonstrate that adversarial attacks pose a real threat to AIGI detectors. FPBA can deliver successful black-box attacks across various detectors, generators, defense methods, and even evade cross-generator and compressed image detection, which are crucial real-world detection scenarios. 

<img width="2496" height="860" alt="image" src="https://github.com/user-attachments/assets/4fe12226-248c-427e-833d-ee97f18d279e" />


## Setup


[Download](https://huggingface.co/collections/Oliver1515/robust-synthetic-image-detector) checkpoints trained on CNNDetect's dataset and example images from huggingface for test.


## Experiments

Attack CNNSpot by FPBA:

```shell
python attack.py \
    --seed 42 \
    --exp_name test \
    --mode attack \
    --bayes True \
    --attack FPBA \
    --batch_size 4 \
    --model CNNSpot \
    --dataset gan \
    --data_root ./gan_exp \
    --ckpt ./checkpoints/CNNSpot_gan.pth \
    --appmodel_ckpt_root ./appended_mlp/CNNSpot \
    --appmodel_ckpt_name _CNNSpot_PYX_AppendedModel_AT.pth \
    --adv_data_path ./output \
    --results_dir ./results

```

Transfer Attack DenseNet or Swin model:

```shell
python test.py \
    --seed 42 \
    --mode tf_atk \
    --exp_name fpba_densenet \
    --earlystop True \
    --bayes False \
    --model DenseNet \
    --dataset gan \
    --data_root ./output \
    --ckpt ./checkpoints/DenseNet_gan.pth \ 
    --results_dir ./results \
    --tf_attack FPBA \
    --surrogate CNNSpot
```

## Cititing FPBA

If you find this paper useful for your research, please use the following BibTeX entry.

```bibtex
@article{diao2024vulnerabilities,
  title={Vulnerabilities in ai-generated image detection: The challenge of adversarial attacks},
  author={Diao, Yunfeng and Zhai, Naixin and Miao, Changtao and Yu, Zitong and Wei, Xingxing and Yang, Xun and Wang, Meng},
  journal={arXiv preprint arXiv:2407.20836},
  year={2024}
}

```
