# CoDoFuzz: Black-Box Co-Domain Coverage Guided Fuzzing of DNNs
The following repository contains code for the paper titled: "Robust Black-box Testing of Deep Neural
Networks using Co-Domain Coverage".

To reproduce paper results, follow given steps sequentially:
- Download datasets

  - Run ```download_datasets.py```. It will download and save the datasets to the folder ```./data```

    ```
    mkdir data
    python  download_datasets.py 
    ```
- Download pre-trained model weights
  - DNN architectures are defined inside the folder ```models``` .
  - Download pre-trained model weights from [this link](https://drive.google.com/drive/folders/1s0i9f3bYhV-TzRN_Qwd7sYR2qDQkBSoX?usp=share_link) and unzip it.
- Run fuzzing using any criteria such as NC, NBC, LBC, SNAC, TKNC, KMNC, LSC, DSC, CC, NLC, and CDC (Co-Domain Coverage).
  - MNIST

    ````
    python fuzz.py --dataset MNIST --model lenet5 --criterion criterion_name  --output_dir results --max_testsuite_size 10000 --seed_id 1
    ````
  
  - FashionMNIST

    ````
    python fuzz.py --dataset FashionMNIST --model LeNet --criterion criterion_name  --output_dir results --max_testsuite_size 10000 --seed_id 1
    ````

  - SVHN

    ````
    python fuzz.py --dataset SVHN --model vgg16 --criterion criterion_name  --output_dir results --max_testsuite_size 10000 --seed_id 1
    ````

  - CIFAR10

    ```
    python fuzz.py --dataset CIFAR10 --model resnet18 --criterion criterion_name  --output_dir results --max_testsuite_size 10000 --seed_id 1
    ```

  - CIFAR100

    ```
    python fuzz.py --dataset CIFAR100 --model resnet34 --criterion criterion_name  --output_dir results --max_testsuite_size 10000 --seed_id 1
    ```
  - ImageNet
    ```
    python fuzz.py --dataset ImageNet --model mobilenet_v2 --criterion criterion_name  --output_dir results --max_testsuite_size 10000 --seed_id 1
    ```
##### Acknowledgement
This repository is build upon the codebase of [Neural-Coverage](https://github.com/Yuanyuan-Yuan/NeuraL-Coverage)  github repository.
