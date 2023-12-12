## Efficient Two-Step Networks for Temporal Action Segmentation
JFormer: Joint-wise Spatio-temporal Transformer for Skeleton-based Action Segmentation [Efficient Two-Step Networks for Temporal Action Segmentation](https://www.sciencedirect.com/science/article/pii/S0925231221006998).

## Requirements
```
* Python 3.8.5
* pyTorch 1.8.1
```

You can download packages using requirements.txt.  
``` pip install -r requirements.txt```

## Datasets
* Download the [data](https://zenodo.org/record/3625992#.Xiv9jGhKhPY) provided by [MS-TCN](https://github.com/yabufarha/ms-tcn),  which contains the I3D features (w/o fine-tune) and the ground truth labels for 3 datasets. (~30GB)
* Extract it so that you have the `data` folder in the same directory as `train.py`.

## directory structure

```
├── config
│   ├── 50salads
│   ├── breakfast
│   └── gtea
├── csv
│   ├── 50salads
│   ├── breakfast
│   └── gtea
├─ dataset ─── 50salads/...
│           ├─ breakfast/...
│           └─ gtea ─── features/
│                    ├─ groundTruth/
│                    ├─ splits/
│                    └─ mapping.txt
├── libs
├── result
├── utils 
├── requirements.txt
├── train.py
├── eval.py
└── README.md
```

## Training and Testing of ETSN
### Setting
First, convert ground truth files into numpy array.
```
python utils/generate_gt_array.py ./dataset
```
Then, please run the below script to generate csv files for data laoder'.
```
python utils/builda_dataset.py ./dataset
```

### Training

You can train a model by changing the settings of the configuration file.
```
python train.py ./config/xxx/xxx/config.yaml
```

### Evaluation
You can evaluate the performance of result after running.
```
python eval.py ./result/xxx/xxx/config.yaml test
```
We also provide trained ETSN model in [Google Drive](https://drive.google.com/drive/folders/1-0k9HVw2XQCXpqXA59kmgS2pQLSHwx87?usp=sharing). Extract it so that you have the `result` folder in the same directory as `train.py`.

### average cross validation results
```
python utils/average_cv_results.py [result_dir]
```
### Citation

If you find our code useful, please cite our paper. 

```
@article{LI2021373,
author = {Yunheng Li and Zhuben Dong and Kaiyuan Liu and Lin Feng and Lianyu Hu and Jie Zhu and Li Xu and Yuhan wang and Shenglan Liu},
journal = {Neurocomputing},
title = {Efficient Two-Step Networks for Temporal Action Segmentation},
year = {2021},
volume = {454},
pages = {373-381},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2021.04.121},
url = {https://www.sciencedirect.com/science/article/pii/S0925231221006998},

}
```
### Contact
For any question, please raise an issue or contact.

### Acknowledgement

We appreciate [MS-TCN](https://github.com/yabufarha/ms-tcn) for extracted I3D feature, backbone network and evaluation code.

Appreciating [Yuchi Ishikawa](https://github.com/yiskw713 ) shares the re-implementation of [MS-TCN](https://github.com/yiskw713/ms-tcn) with pytorch.

