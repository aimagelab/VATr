![test](https://raw.githubusercontent.com/aimagelab/VATr/main/files/model_dark.svg?token=GHSAT0AAAAAACAMD5YD3576YHRB7ZU6UHJCZA4HHHQ#gh-dark-mode-only)
![test](https://raw.githubusercontent.com/aimagelab/VATr/main/files/model_light.svg?token=GHSAT0AAAAAACAMD5YCUSZN2P6EEPSBN3CSZA4HHIQ#gh-light-mode-only)

## Installation

```bash
conda create --name vatr python=3.9
conda activate vatr
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

From [this folder](https://drive.google.com/drive/folders/1FGJe2uCuK8T9HrFzY_Zc-KMIo0oPJGGY?usp=share_link) you have to download the files `IAM-32.pickle` and `resnet_18_pretrained.pth` and place them into the `files` folder.

## Training

```bash
python train.py
```
Useful arguments:
```bash
python train.py
        --feat_model_path PATH  # path to the pretrained resnet 18 checkpoint. If none, the resnet will be trained from scratch
        --dataset DATASET       # dataset to use. Default IAM
        --resume                # resume training from the last checkpoint with the same name
        --wandb                 # use wandb for logging
```

### Pretraining dataset
The model `resnet_18_pretrained.pth` was pretrained using this dataset: [download link](https://drive.google.com/drive/folders/1Xs_rR0EWt09-K6vmlvAI8pwsrmHSknC8?usp=share_link)

## Generate fakes

```bash
python generate_fakes.py --checkpoint files/vatr.pth
```