# Frequency-based Post-train Bayesian Attack

[Download](https://drive.google.com/drive/folders/1wAm8s5dOv9AxlBhIWGOiKLXOV2RGcrbo?usp=sharing) checkpoints trained on CNNDetect's dataset and example images for test.


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
    --appmodel_ckpt_root ./appended_mlp/CNNSpot \
    --appmodel_ckpt_name _CNNSpot_PYX_AppendedModel_AT.pth \
    --adv_data_path ./output \
    --results_dir ./results

```
