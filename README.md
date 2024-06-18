# Multiple Kernel Learning Enhances Relative Positional Encoding Length Extrapolation: The MEP Approach

PyTorch implementation of the paper [Multiple Kernel Learning Enhances Relative Positional Encoding Length Extrapolation: The MEP Approach](https://arxiv.org/abs/2403.17698) . This repository is adapted from the awesome [gpt-neox](https://github.com/EleutherAI/gpt-neox) library.

## Important Changes and Information
1. This repository was developed based on commit 450b58c4ad7f36c319ca0b2f089c7349f34d8c3b of gpt-neox. We bump it to commit 738b87e73775e2cef4ea0a898b655f5d717cb8a0 to include some (irrelevant to this project) bug fixes. We only keep the main branch. (see https://github.com/chijames/KERPLE)
2. We remove the .github/ folder as it is not needed in our experiments.
3. The original gpt-neox readme is renamed as README_gpt_neox.md.
4. The config files used in our experiments are stored in mep_configs/.

## Installation
Please refer to the original readme README_gpt_neox.md for details. We use the Host Setup without fused kernels.

## Data Preparation
Warning: These datasets are huge! Please make sure you have at least **250 GB** of disk space before download them all.

We use the three preconfigured datasets in the orignal gpt-neox repository:
```
python prepare_data.py -d ./data openwebtext2
python prepare_data.py -d ./data arxiv
python prepare_data.py -d ./data github

datasets:
openwebtext2: https://openwebtext2.readthedocs.io
arxiv: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/blob/main/urls/arxiv.txt
github: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/blob/main/urls/github.txt
```
Please refer to the original readme README_gpt_neox.md for details.

## Training
```
bash train.sh
```

## Testing
```
bash test.sh
```

## Main classes
ParallelSNOPE
ParallelSNOPEKerpleLog
