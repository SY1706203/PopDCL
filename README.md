# PopDCL


## Overview

Official code of "PopDCL: Popularity-aware Debiased Contrastive Loss for Collaborative Filtering" (2023 CIKM)


## Run the Code

- We provide implementation for various baselines presented in the paper.

- We also provide the In-Distribution(test_id) and Out-of-Distribution(test_ood) test splits for Amazon-book, Tencent and Alibaba-Ifashion datasets.

- To run the code, first run the following command to install tools used in evaluation:
```
python setup.py build_ext --inplace
```


### LightGCN backbone
For models with LightGCN as backbone, use models with in-batch negative sampling strategy. For example:

- LightGCN Training:

```
python main.py --modeltype LGN --dataset tencent.new --n_layers 2 --neg_sample 1
```

- INFONCE Training:

```
python main.py --modeltype INFONCE_batch --dataset tencent.new  --n_layers 2 --neg_sample -1
```

- BC-LOSS Training:

```
python main.py --modeltype BC_LOSS_batch --dataset tencent.new --n_layers 2 --neg_sample -1
```

- PopDCL Training:
```
python main.py --modeltype PopDCL_LOSS_batch --dataset tencent.new --n_layers 2 --neg_sample -1 --Tau 0.12
```

Details of hyperparameter settings for various baselines can be found in the paper.


## Requirements

- python == 3.7.10

- tensorflow == 1.14

- pytorch == 1.9.1+cu102


## Reference
If you want to use our codes and datasets in your research, please cite:

```
@inproceedings{liu2023popdcl,
  title={PopDCL: Popularity-aware Debiased Contrastive Loss for Collaborative Filtering},
  author={Liu, Zhuang and Li, Haoxuan and Chen, Guanming and Ouyang, Yuanxin and Rong, Wenge and Xiong, Zhang},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={1482--1492},
  year={2023}
}
```

Part of the code comes from https://github.com/anzhang314/BC-Loss










