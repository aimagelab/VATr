# Handwritten Text Generation from Visual Archetypes

This repository contains the reference code and dataset for the paper [Handwritten Text Generation from Visual Archetypes](https://arxiv.org/abs/2303.15269).

[<img alt="alt_text" width="100px" src="https://img.uxwing.com/wp-content/themes/uxwing/download/web-app-development/demo-icon.png"/>](https://vatr-demo.streamlit.app/)

If you find it useful, please cite it as:
```
@inproceedings{pippi2023handwritten,
  title={{Handwritten Text Generation from Visual Archetypes}},
  author={Pippi, Vittorio and Cascianelli, Silvia and Cucchiara, Rita},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

![test](https://github.com/aimagelab/VATr/blob/main/files/model_dark.png?raw=true#gh-dark-mode-only)
![test](https://github.com/aimagelab/VATr/blob/main/files/model_light.png?raw=true#gh-light-mode-only)

## Installation

```console
conda create --name vatr python=3.9
conda activate vatr
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
git clone https://github.com/aimagelab/VATr.git && cd VATr
pip install -r requirements.txt
```

From [this folder](https://drive.google.com/drive/folders/1FGJe2uCuK8T9HrFzY_Zc-KMIo0oPJGGY?usp=share_link) you have to download the files `IAM-32.pickle` and `resnet_18_pretrained.pth` and place them into the `files` folder.

```console
gdown --folder "https://drive.google.com/drive/u/2/folders/1FGJe2uCuK8T9HrFzY_Zc-KMIo0oPJGGY"
```

## Training

```console
python train.py
```

Useful arguments:
```console
python train.py
        --feat_model_path PATH  # path to the pretrained resnet 18 checkpoint. If none, the resnet will be trained from scratch
        --dataset DATASET       # dataset to use. Default IAM
        --resume                # resume training from the last checkpoint with the same name
        --wandb                 # use wandb for logging
```

### Pretraining dataset
The model `resnet_18_pretrained.pth` was pretrained by using this dataset: [download link](https://drive.google.com/drive/folders/1Xs_rR0EWt09-K6vmlvAI8pwsrmHSknC8?usp=share_link)


## Generate styled Handwtitten Text Images

To generate all samples for FID evaluation you can use the following script:

```console
python generate_fakes.py --checkpoint files/vatr.pth
```

To generate a specific text with a given input style folder containing images of handwritten single words you can use the following script:

```console
python generator.py --style-folder "files/style_samples/00" --checkpoint "files/vatr.pth" --output "files/output_00.png" --text "That's one small step for man, one giant leap for mankind ΑαΒβΓγΔδ"
```


Output for `That's one small step for man, one giant leap for mankind ΑαΒβΓγΔδ`:

![test](https://github.com/aimagelab/VATr/blob/main/files/output_00.png?raw=true)


### Implementation details
This work is partially based on the code released for [Handwriting-Transformers](https://github.com/ankanbhunia/Handwriting-Transformers)
