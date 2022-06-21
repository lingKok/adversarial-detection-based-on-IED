import os
import shutil

import torch
from art.attacks.evasion import SaliencyMapMethod, CarliniL0Method,CarliniLInfMethod,CarliniL2Method,DeepFool,ProjectedGradientDescentPyTorch,AdversarialPatch,FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from torch import nn

from Attacks.AttackMethods.AttackUtils import predict
from RawModels.MNISTConv import MNISTConvNet
from RawModels.ResNet import resnet20_cifar
from RawModels.Resnet_for_imagenet import resnet34_imagenet
from RawModels.ResNet_for_stl10 import resnet20_stl
from RawModels.SVHNConv import SVHNConvNet
import numpy as np

# os.makedirs('./AdversarialExampleDatasets/CWL2/IMAGENET10',exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_list = ['MNIST']#['MNIST','CIFAR10','SVHN','IMAGENET10']
loss_func = nn.CrossEntropyLoss()
batch_size = 10
attack_list = ['CWLINF']#['DEEPFOOL','FGSM','PGD','JSMA','CWL2','CWL0','CWLINF']#,
for dataset in dataset_list:


    for attack_name in attack_list:
        adv_examples_dir = './AdversarialExampleDatasets/'+attack_name+'/'+dataset+'/'
        clean_data_location = './CleanDatasets/'
        raw_model_location='./RawModels/'
        raw_model_location = '{}{}/model/{}_raw.pt'.format(
                    raw_model_location, dataset, dataset)
        if dataset == 'MNIST' or dataset == 'FSMNIST':
            raw_model = MNISTConvNet().to(device)
            raw_model.load(path=raw_model_location, device=device)
            raw_model = PyTorchClassifier(raw_model, loss=loss_func, input_shape=(1, 28, 28), nb_classes=10, optimizer=None)
        if dataset == 'SVHN':
            raw_model = SVHNConvNet().to(device)
            raw_model.load(path=raw_model_location, device=device)
            raw_model = PyTorchClassifier(raw_model, loss=loss_func, input_shape=(3, 32, 32), nb_classes=10, optimizer=None)

        if dataset == 'STL10':
            raw_model = resnet20_stl().to(device)
            raw_model.load(path=raw_model_location, device=device)
            raw_model = PyTorchClassifier(raw_model, loss=loss_func, input_shape=(3, 32, 32), nb_classes=10, optimizer=None)

        if dataset == 'CIFAR10':
            raw_model = resnet20_cifar().to(device)
            raw_model.load(path=raw_model_location, device=device)
            raw_model = PyTorchClassifier(raw_model, loss=loss_func, input_shape=(3, 32, 32), nb_classes=10, optimizer=None)
        if dataset == 'IMAGENET10':
            raw_model = resnet34_imagenet().to(device)
            raw_model.load_state_dict(torch.load(raw_model_location))
            raw_model = PyTorchClassifier(raw_model, loss=loss_func, input_shape=(3, 224, 224), nb_classes=10, optimizer=None)



        # get the clean data sets / true_labels / targets (if the attack is one of the targeted attacks)
        print(
            'Loading the prepared clean samples (nature inputs and corresponding labels) that will be attacked ...... '
        )
        nature_samples = np.load('{}{}/{}_inputs.npy'.format(
            clean_data_location, dataset, dataset))
        labels_samples = np.load('{}{}/{}_labels.npy'.format(
            clean_data_location, dataset, dataset))
        if attack_name =='JSMA':
            # raw_model = PyTorchClassifier(raw_model,loss=None,input_shape=(3,32,32),nb_classes=10,optimizer=None)
            # # raw_model.set_params()
            jsma = SaliencyMapMethod(raw_model)
            adv_samples=jsma.generate(nature_samples)
        if attack_name == 'CWL0':
            cwl0 = CarliniL0Method(raw_model)
            iteration = int(np.ceil(len(nature_samples) / float(batch_size)))
            adv_samples = np.array([])
            for i in range(iteration):
                start = i * 10
                end = np.minimum((i + 1) * 10, len(nature_samples))
                adv_sample = cwl0.generate(nature_samples[start:end])
                if adv_samples.size == 0:
                    adv_samples = adv_sample
                else:
                    adv_samples = np.concatenate((adv_samples, adv_sample))


        if attack_name == 'CWLINF':
            cwlinf = CarliniLInfMethod(raw_model)
            iteration = int(np.ceil(len(nature_samples) / float(batch_size)))
            adv_samples = np.array([])
            for i in range(iteration):
                start = i * 10
                end = np.minimum((i + 1) * 10, len(nature_samples))
                adv_sample = cwlinf.generate(nature_samples[start:end])
                if adv_samples.size == 0:
                    adv_samples = adv_sample
                else:
                    adv_samples = np.concatenate((adv_samples, adv_sample))

        if attack_name == 'CWL2':
            cwl2 = CarliniL2Method(raw_model)
            iteration = int(np.ceil(len(nature_samples)/float(batch_size)))
            adv_samples = np.array([])
            for i in range(iteration):
                start = i*10
                end = np.minimum((i+1)*10,len(nature_samples))
                adv_sample = cwl2.generate(nature_samples[start:end])
                if adv_samples.size==0:
                    adv_samples = adv_sample
                else:
                    adv_samples = np.concatenate((adv_samples,adv_sample))

        if attack_name == 'PGD':
            pgd = ProjectedGradientDescentPyTorch(raw_model,eps=0.1)
            adv_samples = pgd.generate(nature_samples)
        if attack_name == 'FGSM':
            fgsm = FastGradientMethod(raw_model,eps=0.3)
            adv_samples = fgsm.generate(nature_samples)
        if attack_name == 'DEEPFOOL':
            dp = DeepFool(raw_model)
            adv_samples = dp.generate(nature_samples)
        # cwl0 = CarliniL0Method(raw_model)
        # adv_samples = cwl0.generate(nature_samples)
        adv_labels = raw_model.predict(adv_samples)
        # adv_labels = np.load('{}{}_AdvLabels.npy'.format(adv_examples_dir, attack_name))
        adv_labels = torch.from_numpy(adv_labels)
        adv_labels = torch.max(adv_labels, 1)[1]
        adv_labels = adv_labels.cpu().numpy()
        if not os.path.isdir(adv_examples_dir):
            os.makedirs(adv_examples_dir,exist_ok=True)
        np.save('{}{}_AdvExamples.npy'.format(adv_examples_dir, attack_name), adv_samples)
        np.save('{}{}_AdvLabels.npy'.format(adv_examples_dir, attack_name), adv_labels)
        np.save('{}{}_TrueLabels.npy'.format(adv_examples_dir, attack_name), labels_samples)