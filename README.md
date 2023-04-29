[![DOI](https://zenodo.org/badge/631492095.svg)](https://zenodo.org/badge/latestdoi/631492095)

# easyFL: A Lightning Framework for Federated Learning

This code is PyTorch implementation for our paper.

The repository is a modified version of [easyFL](https://github.com/WwZzz/easyFL), which was introduced by the authors in **Federated Learning with Fair Averaging** (IJCAI-21). EasyFL is a strong and reusable experimental platform for research on federated learning (FL) algorithm. It is easy for FL-researchers to quickly realize and compare popular centralized federated learning algorithms. 

## Requirements

The project is implemented using Python3 with dependencies below:

```
numpy=1.22.3
python=3.10.6
pytorch=1.13.1
torchvision=0.14.1
scipy=1.9.3
matplotlib=3.5.2
wandb=0.12.21
pandas=1.4.3
pyyaml=6.0
scikit-learn=1.1.1
seaborn=0.12.1
```
You may create a environment from environment.yml
## QuickStart

To run the experiments, simply:
Run any bash file in folder /scripts

**Note**: 

LOG_DIR: where to log results  
DATA_DIR: location of dataset CIFAR10 or MNIST  
ALG: single or multi-process algorithm  
MODEL: DL model to train  
DIRTY_RATE: list of poisoned sample ratio of all clients  
NOISE_MAGNITUDE: magnitude of noise  
NOISE_TYPE: noise type is one of {'gaussian', 'poisson', 'salt_peper', 'speckle'}  
MALICIOUS_CLIENT: the number of malicious clients  
ATTACKED_CLASS: the classes is poissoned  
AGGREGATE and AGG_ALGORITHM to specify the algorithm used  
WANDB: 1 if use wandb otherwise 0  
ROUND: the number of rounds to train  
EPOCH_PER_ROUND: the number of local epochs  
BATCH: batch size  
PROPOTION: the ratio of participating clients in each round  
NUM_THRESH_PER_GPU: the number of thresh per GPU device  
NUM_GPUS: the number of GPU devices  
SERVER_GPU_ID: the index of server GPU  
TASK: the name of task  
IDX_DIR: location of splitted dataset for all clients  

To run experiments, you must define arguments in bash file then run that bash file.  
If you do not have a wandb account, you can set "WANDB=0" in bash file.  
If you want to change noise type i.e. poisson noise, you can set NOISE_TYPE=’poisson’ in bash file.  
If you want to run Fedavg, you can set AGGREGATE=’mean’ and AGG_ALGORITHM="fedavg" in bash file.  
If you want to run our defend NDAD, you can set AGGREGATE=’mean’ and AGG_ALGORITHM="NDAD" in bash file.  
If you want to run other defenses i.e. median, you can set AGGREGATE=’median’ and AGG_ALGORITHM="median" in bash file.  

Run the bash files to reproduce the results.  
The results will be stored in the folder specified in “LOG_DIR” which will be automatically created when running the bash file. If you use wandb, the results will be stored in wandb website.  

