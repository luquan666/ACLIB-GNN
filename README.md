# ACLIB-GNN：Incorporating Adversarial Causal Learning with Information Bottlenecks for Interpretable Graph Neural Networks
This is the official code for the implementation of "ACLIB-GNN：Incorporating Adversarial Causal Learning with Information Bottlenecks for Interpretable Graph Neural Networks"
## Table of contents
* [Overview](#overview)
* [Installation](#installation)
* [Run ACLIB-GNN](#run-leci)
* [Citing LECI](#citing-leci)
* [License](#license)
* [Contact](#contact)

## Overview

In this work, we propose to simultaneously incorporate label and environment causal independence (LECI) to 
release the potential of pre-collected environment information in graph tasks, thereby addressing the challenges faced by prior methods on identifying 
causal/invariant subgraphs. We further develop an adversarial training strategy to jointly optimize these two properties for 
causal subgraph discovery with theoretical guarantees.

## Installation
```shell
conda create -n ACLIB-GNN python=3.9
conda activate ACLIB-GNN
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_cluster-1.6.3%2Bpt25cu121-cp39-cp39-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_scatter-2.1.2%2Bpt25cu121-cp39-cp39-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_sparse-0.6.18%2Bpt25cu121-cp39-cp39-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_spline_conv-1.2.2%2Bpt25cu121-cp39-cp39-win_amd64.whl
pip install torch-geometric==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas==2.2.3
pip install numpy==1.26.4
pip install scikit-learn==1.6.1
```
