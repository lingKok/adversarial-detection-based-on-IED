# Adversarial Detection based on Inner-class Euclidean Distance


## 1. Description
Adversarial detection based on inner-class Euclidean distance is a method to detect adversarial examples.



### 1.1 Glance at the Repo:

- `RawModels/` contains code used to train models and trained models will be attacked;
- `CleanDatasets/` contains code used to randomly select clean samples to generate adversarial examples.

### 1.3 Requirements:

Make sure you have installed all of following packages or libraries (including dependencies if necessary) in you machine:

1. PyTorch 0.4
2. TorchVision 0.2
3. numpy, scipy, PIL, skimage, tqdm ...

### 1.4 Datasets:
We mainly employ two benchmark dataset MNIST,SVHN and CIFAR10.


## 2. Usage/Experiments


### STEP 1. Training the raw models and preparing the clean samples
We firstly train and save the deep learning models for MNIST and CIFAR10 [here](./RawModels/), and then randomly select and save the clean samples that will be attacked [here](./CleanDatasets/).

### STEP 2. Generating adversarial examples
Taking the trained models and the clean samples as the input, we can generate corresponding adversarial examples for each kinds of adversarial attacks that we have implemented in art including FGSM,PGD,JSMA,DEEPFOOL,CW$_0$,CW$_2$,CW$_\infty$.
.
>python adv_generator.py 

### STEP 3. Detect adversarial examples
####  supervised method
>python detect_train.py --dataset MNIST -- attack FGSM --type ied
#### unsupervised method
>python unsupervised_detect.py --dataset MNIST --attack FGSM -- type ied
