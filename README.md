[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tnt-target-driven-trajectory-prediction/trajectory-prediction-on-interaction-dataset-2)](https://paperswithcode.com/sota/trajectory-prediction-on-interaction-dataset-2?p=tnt-target-driven-trajectory-prediction)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tnt-target-driven-trajectory-prediction/motion-forecasting-on-argoverse-cvpr-2020)](https://paperswithcode.com/sota/motion-forecasting-on-argoverse-cvpr-2020?p=tnt-target-driven-trajectory-prediction)

# TNT-Trajectory-Predition

A Python and Pytorch implementation of 
[TNT: Target-driveN Trajectory Prediction](https://arxiv.org/abs/2008.08294#:~:text=TNT%20has%20three%20stages%20which,state%20sequences%20conditioned%20on%20targets.)
and
[VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/abs/2005.04259).

> **ATTENTION**: Currently, the training consumes a large memory (128G+) since an in-memory data loader is implemented. We'll provide a new dataloader that loading only training batches from disk for each step in the coming future.


### Achieved Best Performance
The best performance achieved by our implementation and reported in the papers on the evaluation dataset. 

| Algorithm | minADE(K=1) | minFDE(K=1) | MR(2.0m) | minADE(K=6) | minFDE(K=6) | MR(2.0m) |
| :-------: |:-----------:|:-----------:|:--------:|:-----------:|:-----------:|:--------:|
| VectorNet (Original) |    1.66     |    3.67     |    -     |      -      |      -      |    -     |
| VectorNet (**Ours**) |    1.46     |    3.15     |  0.548   |      -      |      -      |    -     |
|  TNT(Original)   |      -      |      -      |    -     |    0.728    |    1.292    |  0.093   |
|  TNT(**Ours**)   |      -      |      -      |    -     |    0.928    |    1.686    |  0.195   |

The best performance achieved by our implementation and reported in the papers on the test dataset on 
[Argoverse Leaderboard](https://eval.ai/web/challenges/challenge-page/454/leaderboard).

|    Algorithm     |  minADE(K=6)  | minFDE(K=6)  | MR(2.0m)  |
|:----------------:|:-------------:|:------------:|:---------:|
| 	TNT - CoRL20    |    0.9097     |    1.4457    |  0.1656   |
|  	CUHK RPAI (TNT_20220819)   |    1.1518     |    2.1322    |  0.2585   |


* [VectorNet Pre-trained Weights](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155071948_link_cuhk_edu_hk/Ee7FZXGwXB9Mh7O7wSHjAlUBH5uB1fP9LEXPP8TS1lSFTQ?e=ajPXMo)
* [TNT Pre-trained Weights](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155071948_link_cuhk_edu_hk/EfgqFbzKKJlJle7E-y-DVCkBVQQQH09CUwTDN5GfjtYAOg?e=mUINGD)

## Setup

### Prerequisite
This implementation has been tested on Ubuntu 18.04 and has the following requirements:
* python == 3.8.8
* pytorch == 1.8.1
* torch-geometric == 1.7.2 (The version of related libraries are listed as follows.)
  * pytorch-cluster == 1.5.9          
  * pytorch-geometric == 1.7.2           
  * pytorch-scatter == 2.0.7           
  * pytorch-sparse == 0.6.10         
  * pytorch-spline-conv == 1.2.1
* pandas == 1.4.4
* tqdm == 4.60.0
* tensorboard
* (Optianl) [nvidia-apex](https://github.com/NVIDIA/apex) == 1.0

* [Argoverse-api](https://github.com/argoai/argoverse-api)

You can also install the dependencies using pip.
```
pip install -r requirements.txt
```

### Data Preparation
Please download the [Argoverse Motion Forecasting v1.1](https://www.argoverse.org/av1.html#forecasting-link) and extract 
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

An example of the processed dataset(small) is available 
[here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155071948_link_cuhk_edu_hk/EWv6Dec3yqhPkZLgdw9eEt0Bso2K-ef1-7UOzklwc4NDPQ?e=2TIBGX).
Download it to check if your environment is configured appropriately. 

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

4. For users with multiple GPUs, this implementation uses NVIDIA Apex library to enable the distributed parallel training. To enable the multi-gpu training:
   ```
   python -m torch.distributed.launch --nproc_per_node=2 train_tnt.py ...
   ```
   or enable corresponding command in the bash scripts for training. 
### Inference

A python script named "test_tnt.py" is provided for the inference of TNT model. 
It can make prediction on the test set and generate .h5 files required by the Argoverse benchmark. 
You need to specify the file path to your trained model file or checkpoint.
Since the model are trained in the relative coordinate, a coordinate conversion has been enabled to recover the 
world coordinates of the predicted trajectories.

> **Reminding**: Specify either the file path to the checkpoint or the trained model. 
> Also, specify the path to your dataset if you are using a different directory structure.

The prediction result can be genearted withe command:
```
python test_tnt.py -rm Path_to_Your_Model_File
```

## Others

**TBD**

### TODO
1. Data-Related:
- [x] Preprocessing of test set;

2. Model-Related:
- [ ] Create a base class for models;

3. Training-Related:
- [x] Enable multi-gpu training; (Using Nvidia APEX library, will be merged to main branch later...)
- [ ] Enable loading data from the hard disk. 

4. Inference-Related:
- [x] Provide the inference function to visualize the input sequences and corresponding results.

### Citing


if you've found this code to be useful, please consider citing our paper!
```
Liu, J., Mao, X., Fang, Y., Zhu, D., & Meng, M. Q. H. (2021). A Survey on Deep-Learning Approaches for Vehicle Trajectory Prediction in Autonomous Driving. arXiv preprint arXiv:2110.10436.

```

### Questions

This repo is maintained by [Jianbang Liu](henryliu@link.cuhk.edu.hk) and [Xinyu Mao](maoxinyu@link.cuhk.edu.hk) - please feel free to reach out or open an issue if you have additional questions/concerns.
