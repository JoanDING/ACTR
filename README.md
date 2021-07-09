# This is the repo for the paper: [Modeling Instant User Intent and Content-level Transition for Sequential Fashion Recommendation] 

## Requirements
1. OS: Ubuntu 16.04 or higher version
2. python3.7
3. Supported (tested) CUDA Versions: V10.2
4. python modules: refer to the modules in [requirements.txt](https://github.com/JoanDING/ACTR_Pub/blob/main/requirements.txt)


## Code Structure
1. The entry script for training and evaluation is: [train.py](https://github.com/JoanDING/ACTR_Pub/blob/main/train.py)
2. The config file is: [config.yaml](https://github.com/JoanDING/ACTR_Pub/blob/main/config.yaml)
3. The script for data preprocess and dataloader: [utility.py](https://github.com/JoanDING/ACTR_Pub/blob/main/utility.py)
4. The model script is [ACTR.py](https://github.com/JoanDING/ACTR_Pub/blob/main/ACTR.py)
5. The experimental logs in tensorboard-format are saved in ./logs 
6. The experimental logs in txt-format are saved in ./performance.
7. The recommendation results in the evaluation are recorded in ./results.
8. ./logs, ./performance, ./results will be generated automatically for the first-time runing. 


## How to Run
1. Download the [dataset](https://drive.google.com/file/d/1Jtxwu5vJzv2JFmlGGJA-H-N52PxV57J2/view?usp=sharing), decompress it and put it in the top directory with the following command. Note that the downloaded files include two datasets ulilized in the paper: iFashion and Amazon_Fashion.
    ```
    tar zxvf actr_dataset.tar.gz
    ```

2. Settings in the configure file config.yaml are basic experimental settings, which are usually fixed in the experiments. To tune other hyper-parameters, you can use command line to pass the parameters. The command line supported hyper-parameters including: the dataset (-d), sequence length (-l). You can also specify which gpu device (-g) to use in the experiments and choose how many times to repeat the experiment by (-r).

3. You can specify the hyper-parameters for each experiment. For example, to train and evaluate the ACTR on the ifashion dataset with the sequence length being 5, you can use the following command: 
    ```
    python train.py -d=ifashion -l=5
    ```

4. During the training, you can monitor the training loss and the evaluation performance by Tensorboard. You can get into ./logs to track the curves of your training and evaluation with the following command:
    ```
    tensorboard --host="your host ip" --logdir=./
    ```

5. The performance of the model is saved in ./performance. You can get into the folder and check the detailed training process of any finished experiments (Compared with the tensorboard log save in ./logs, it is just the txt-version human-readable training log). 
