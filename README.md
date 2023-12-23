## MGLNN
The official implementation of paper "[MGLNN: Semi-supervised learning via Multiple Graph Cooperative Learning Neural Networks](https://doi.org/10.1016/j.neunet.2022.05.024)" published in Neural Networks 2022.

## Requirements
* Python = 3.7
* Pytorch = 1.8.1
* Cuda = 10.1

## Introduction
In this repo, we provide MGLNN's code with the Caltech101_7, Reuters and WebKB datasets as example, where the code is implemented based on [Graph Structure Learning for Robust Graph Neural Networks](https://dl.acm.org/doi/abs/10.1145/3394486.3403049) (ProGNN, KDD 2020).

## Examples
* To run the training and evaluation, one can use the scripts in run.sh: 
```
sh run.sh
```

## Cite
Please cite our paper if you use this code in your own work:

```
@article{jiang2022mglnn,
  title={MGLNN: Semi-supervised learning via multiple graph cooperative learning neural networks},
  author={Jiang, Bo and Chen, Si and Wang, Beibei and Luo, Bin},
  journal={Neural Networks},
  volume={153},
  pages={204--214},
  year={2022},
  publisher={Elsevier}
}
```
