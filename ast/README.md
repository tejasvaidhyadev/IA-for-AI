
# AST: Audio Spectrogram Transformer  
 - [Introduction](#Introduction)
 - [Getting Started](#Getting-Started)
 - [ESC-50 Recipe](#ESC-50-Recipe)  
 - [Speechcommands Recipe](#Speechcommands-V2-Recipe)  
 - [AudioSet Recipe](#Audioset-Recipe)
 - [Pretrained Models](#Pretrained-Models)
 - [Use Pretrained Model For Downstream Tasks](#Use-Pretrained-Model-For-Downstream-Tasks)

## Introduction

<p align="center"><img src="https://github.com/tejasvaidhyadev/IA-for-AI/blob/main/ast/ast.png?raw=true" alt="Illustration of AST." width="300"/></p>

AST is the first **convolution-free, purely** attention-based model for audio classification which supports variable length input and can be applied to various tasks.

The AST model file is in `src/models/ast_models.py`, the recipe is in `egs/hack/run.sh`, when you run `run.sh`, it will call `/src/run.py`, which will then call `/src/dataloader.py` and `/src/traintest.py`, which will then call `/src/models/ast_models.py`.

  
## Getting Started  

Step 1: Clone or download this repository, replicate the environment as mentioned in README and install the dependencies. Set this folder as the working directory.

Step 2: Download the datasets inside `ast/egs/hack/data`

Step 3: Prepare the dataset from inside `ast/egs/hack`: `python3 prep_hack.py` for training and `python3 prep_test.py` for test

Step 4. Train the AST model: `bash run.sh`. You may edit following Parameters: inside the file:
- `label_dim` : The number of classes (default:`527`).
- `fstride`:  The stride of patch spliting on the frequency dimension, for 16\*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6 (default:`10`)
- `tstride`:  The stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6 (default:`10`)
-`input_fdim`: The number of frequency bins of the input spectrogram. (default:`128`)
- `input_tdim`: The number of time frames of the input spectrogram. (default:`1024`, i.e., 10.24s)
- `imagenet_pretrain`: If `True`, use ImageNet pretrained model. (default: `True`, we recommend to set it as `True` for all tasks.)
- `audioset_pretrain`: If`True`,  use full AudioSet And ImageNet pretrained model. Currently only support `base384` model with `fstride=tstride=10`. (default: `False`, we recommend to set it as `True` for all tasks)
- `model_size`: The model size of AST, should be in `[tiny224, small224, base224, base384]` (default: `base384`).

**Input:** Tensor in shape `[batch_size, temporal_frame_num, frequency_bin_num]`.
**Output:** Tensor of raw logits (i.e., without Sigmoid) in shape `[batch_size, label_dim]`.

If required, rather than training from scratch, you may set the following in `run.sh`.:
- Imagenet Pretraining: `imagenetpretrain=True`
- Audio Pretraining: `audiosetpretrain=True`

Following are the links to Audiosetpretrained models:
1. [(10, 10, Weight Averaging)](https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1)
2. [(10, 10)](https://www.dropbox.com/s/1tv0hovue1bxupk/audioset_10_10_0.4495.pth?dl=1)
5. [(12, 12)](https://www.dropbox.com/s/snfhx3tizr4nuc8/audioset_12_12_0.4467.pth?dl=1)
6. [(14, 14)](https://www.dropbox.com/s/z18s6pemtnxm4k7/audioset_14_14_0.4431.pth?dl=1)
7. [(16, 16)](https://www.dropbox.com/s/mdsa4t1xmcimia6/audioset_16_16_0.4422.pth?dl=1)

Audio models defaults to first one (10, 10, Weight averaging) when you set audio pretaining to True.

Note: the above dropbox links can be downloaded by wget.

## References

[YuanGongND/ast](https://github.com/YuanGongND/ast)
