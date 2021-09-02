[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tnt-target-driven-trajectory-prediction/trajectory-prediction-on-interaction-dataset-2)](https://paperswithcode.com/sota/trajectory-prediction-on-interaction-dataset-2?p=tnt-target-driven-trajectory-prediction)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tnt-target-driven-trajectory-prediction/motion-forecasting-on-argoverse-cvpr-2020)](https://paperswithcode.com/sota/motion-forecasting-on-argoverse-cvpr-2020?p=tnt-target-driven-trajectory-prediction)

# TNT-Trajectory-Predition

A Python and Pytorch implementation of 
[TNT: Target-driveN Trajectory Prediction](https://arxiv.org/abs/2008.08294#:~:text=TNT%20has%20three%20stages%20which,state%20sequences%20conditioned%20on%20targets.)
and
[VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259).

### Achieved Best Performance
The best performance achieved by our implementation and reported in the papers. 

| Algorithm | minADE(K=1)  | minFDE(K=1) | MR(2.0m) | minADE(K=6) | minFDE(K=6) | MR(2.0m) |
| :-------: | :-----------:| :---------: | :-------:| :----------:| :---------: | :-------:|
| VectorNet (Original) | 1.66        | 3.67       | -        | -           | -           | -        |
| VectorNet (**Ours**) | 1.855       | 3.845      | 0.649    | -           | -           | -        |
|  TNT(Original)   | -            | -           | -        | 0.728       | 1.292       | 0.093    |
|  TNT(**Ours**)   | -            | -           | -        | 1.138       | 2.123       | 0.286    |

## Setup
***

### Prerequisite
This implementation has been tested on Ubuntu 18.04 and has the following requirements:
* python==3.8.8
* pytorch==1.8.1
* torch-geometric==1.7.2
* pandas==1.0.0
* tqdm==4.60.0
* tensorboard

* [Argoverse-api](https://github.com/argoai/argoverse-api)

You can also install the dependencies using pip.
```
pip install -r requirements.txt
```
### Data Preparation
Please download the [Argoverse Motion Forecasting v1.1](https://www.argoverse.org/data.html#forecasting-link) and extract 
the compressed data to the "raw_datas" folder as stored in the following directory structure:
```latex
├── TNT-Trajectory-Prediction
│   ├── dataset
│   │   ├── raw_data
│   │   │   ├── train
│   │   │   │   ├── *.csv
│   │   │   ├── val
│   │   │   │   ├── *.csv
│   │   │   ├── test_obs
│   │   │   │   ├── *.csv
│   │   ├── interm_data
```
Then, direct to the root directory of this code implementation in terminal (``/path_you_place_this_folder/TNT-Trajectory-Prediction``) and run the command:
```
./scripts/preprocessing.sh
```
If you store the raw data at a different location, you can change relative path in the bash script with your path. 

> **ATTENTION**: If you aren't familiar with the bash script and path routing, just follow the directory structure.

> **Reminding**: Change the mode of the bash file with "chmod +x scripts/preprocessing.sh" before running the script for the first time.

## Usage

### Training
1. To train the VectorNet model using NVIDIA GPU and CUDA:
    ```
    python train_vectornet.py --data_root dataset/interm_data --output_dir run/tnt/ \
                              --aux_loss --batch_size 64 --with_cuda --cuda_device 0 \
                              --lr 0.0010 --warmup_epoch 30 --lr_update_freq 10 --lr_decay_rate 0.1
    ```
    or run the bash script: ``./scripts/train_vectornet.sh``.
    > **Reminding**: Change the mode of the bash file with "chmod +x scripts/train_vectornet.sh" before running the script for the first time.

2. To train the TNT model using NVIDIA GPU and CUDA:
    ```
    python train_tnt.py --data_root dataset/interm_data --output_dir run/tnt/ \
                        --aux_loss --batch_size 64 --with_cuda --cuda_device 0 \
                        --lr 0.0010 --warmup_epoch 30 --lr_update_freq 10 --lr_decay_rate 0.1
    ```
    or run the bash script: ``./scripts/train_tnt.sh``. 
    > **Reminding**: Change the mode of the bash file with "chmod +x scripts/train_vectornet.sh" before running the script for the first time.

3. For more configuretion, please refer to the parsers in ``train_tnt.py`` and ``train_vectornet.py``.

### Inference

**TBD**

## Others


### TODO
1. Data-Related:
-[ ] Preprocessing of test set;

2. Model-Related:
- [ ] Create a base class for models;

3. Training-Related:
- [x] Implement the Gaussian negative log-likelihood loss function (Currently, mse loss);
- [ ] Enable multi-gpu training;
- [ ] Enable loading data from the hard disk. 

4. Inference-Related:
- [ ] Provide the inference function to visualize the input sequences and corresponding results.

### Citing


if you've found this code to be useful, please consider citing our paper!
```
xxx

```

### Questions

This repo is maintained by [Jianbang Liu](henryliu@link.cuhk.edu.hk) and [Xinyu Mao](maoxinyu@link.cuhk.edu.hk) - please feel free to reach out or open an issue if you have additional questions/concerns.
