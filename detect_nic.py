from __future__ import division, absolute_import, print_function

import argparse
import os
import pickle
import time
import random

import torch
from sklearn import svm
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sklearn.decomposition import PCA
import numpy as np
# from tensorflow import optimizers
# from tensorflow.python.keras import *
# from tensorflow.python.keras.layers import *
# from tensorflow.python.keras.losses import categorical_crossentropy
# from thundersvm import OneClassSVM

# from common.util import *
# from setup_paths import *
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.svm import OneClassSVM
from torch.nn import Sequential,Linear,Softmax
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN, CIFAR10
from art.estimators.classification import PyTorchClassifier

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')



def dense(input_shape,args):
    model = Sequential(Linear(input_shape,10),
                       Softmax(dim=1))
    return model


def train(model, data, label, args):
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim import Adam
    import torch.nn.functional as F
    print('starting training!')
    optimizer = Adam(params=model.parameters(), lr=0.001)

    data = torch.from_numpy(data)


    label = torch.from_numpy(np.array(label))
    print(data.size(0))
    # print(label.size(0))
    dataset = TensorDataset(data, label)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    model.to(device)
    for i in range(args.epochs):

        for j, data in enumerate(loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = F.cross_entropy(output, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j%100 ==0:
                pred = output.max(1)[1]
                correct = pred.eq(labels.view_as(pred)).sum()
                acc = correct/pred.size(0)
                print('ACC is {:.4f}%'.format(acc*100))

    print('finished training!')


def get_layers_feature(model,data_loader,nb_layer,device,rand):

    import random
    layers_feature = {}
    label = np.array([])
    for data in data_loader:
        x, y = data
        x, y = x.to(device), y.to(device)
        output = model.model(x)
        cor_ind = torch.argmax(output, 1).eq(y).cpu().numpy()
        x = x.cpu().numpy()
        # print(type(x))
        for i in range(nb_layer):
            if layers_feature.get(i, None) is None:
                tem= model.get_activations(x, i).reshape(len(x),-1)
                # print(type(tem))
                layers_feature[i]=tem[cor_ind]


            else:
                tem = model.get_activations(x, i).reshape(len(x), -1)
                layers_feature[i] = np.concatenate((layers_feature[i],tem[cor_ind]
                                                    ))
        if label.shape == 0:
            label = y[cor_ind].cpu().numpy()
        else:
            label = np.concatenate((label, y[cor_ind].cpu().numpy()))
    if rand!=0:
        ind = random.sample(range(len(label)),rand)
        for i in range(nb_layer):
            layers_feature[i]= layers_feature[i][ind]
        label = label[ind]

    return layers_feature,label
# def adv_loader(type, dataset, batch_size):
#     from torch.utils.data import TensorDataset, DataLoader
#     advdata_path = './AdversarialExampleDatasets/' + type + '/' + dataset + '/' + type + '_AdvExamples.npy'
#     Advlabel_path = './AdversarialExampleDatasets/' + type + '/' + dataset + '/' + type + '_AdvLabels.npy'
#     adv_data = np.load(advdata_path)
#     adv_label = np.load(Advlabel_path)
#     adv_data = torch.from_numpy(adv_data)
#     adv_label = torch.from_numpy(adv_label)
#     adv_data = adv_data.to(torch.float32)
#     # _, true_label = torch.max(true_label, 1)
#     # print(true_label.shape)
#     adv_dataset = TensorDataset(adv_data, adv_label)
#     adv_loader = DataLoader(adv_dataset, batch_size=batch_size)
#     return adv_loader
def adv_loader(attack, dataset, batch_size):
    from torch.utils.data import TensorDataset, DataLoader
    advdata_path = './AdversarialExampleDatasets/{}/{}/{}_AdvExamples.npy'.format(attack, dataset, attack)
    Advlabel_path = './AdversarialExampleDatasets/{}/{}/{}_AdvLabels.npy'.format(attack, dataset, attack)
    Turelabel_path = './AdversarialExampleDatasets/{}/{}/{}_TrueLabels.npy'.format(attack, dataset, attack)
    adv_data = np.load(advdata_path)
    adv_label = np.load(Advlabel_path)
    true_label = np.argmax(np.load(Turelabel_path),1)

    cor_ind = np.not_equal(adv_label,true_label)
    print(cor_ind)
    adv_data = torch.from_numpy(adv_data[cor_ind])
    adv_label = torch.from_numpy(adv_label[cor_ind])
    adv_data = adv_data.to(torch.float32)
    # _, true_label = torch.max(true_label, 1)
    # print(true_label.shape)
    adv_dataset = TensorDataset(adv_data, adv_label)
    adv_loader = DataLoader(adv_dataset, batch_size=batch_size)
    print('The successful attack number is :{}'.format(len(adv_label)))
    return adv_loader
def process(Y):
    for i in range(len(Y)):
        if Y[i] == 1:
            Y[i] = 1
        elif Y[i] == -1:
            Y[i] = 0
    return Y

def get_tpr_fpr(true_labels, pred_labels):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

    AP = np.sum(true_labels)
    AN = np.sum(1-true_labels)

    tpr = TP/AP if AP>0 else np.nan
    fpr = FP/AN if AN>0 else np.nan

    return tpr, fpr, TP, AP, FP, AN
def evalulate_detection_test(Y_detect_test, Y_detect_pred):
    accuracy = accuracy_score(Y_detect_test, Y_detect_pred, normalize=True, sample_weight=None)
    tpr, fpr, tp, ap, fp, an = get_tpr_fpr(Y_detect_test, Y_detect_pred)
    return accuracy, tpr, fpr, tp, ap, fp, an
def main():


    print('Loading the data and model...')
    # Load the model
    import argparse
    parse = argparse.ArgumentParser(description='this is an adversarial detection based on Local Cosine Similarity')
    parse.add_argument('--seed', type=int, default=100, help='set the random seed')
    parse.add_argument('--gpu_index', type=str, default='1', help='the index of gpu')
    parse.add_argument('--dataset', default='SVHN', choices=['MNIST', 'SVHN', 'CIFAR10'])
    parse.add_argument('--attack', default='FGSM', choices=['FGSM', 'PGD', 'DEEPFOOL', 'JSMA', 'CWL0','CWL2','CWLINF'])
    parse.add_argument('--test_attack', default='FGSM', choices=['FGSM', 'PGD', 'DEEPFOOL', 'JSMA', 'CW2'])
    parse.add_argument('--batch_size', type=int, default=64)
    parse.add_argument('--epochs', type=int, default=10)
    args = parse.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # Set the random seed manually for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dataset == 'MNIST':
        from RawModels.MNISTConv import MNISTConvNet
        rawModel_path = 'RawModels/MNIST/model/MNIST_raw.pt'
        train_data = MNIST(root='./RawModels/MNIST/', train=True, transform=transforms.ToTensor())
        test_data = MNIST(root='./RawModels/MNIST/', train=False, transform=transforms.ToTensor())
        X_train = train_data.data
        Y_train = train_data.targets
        X_test = test_data.data
        y_test = test_data.targets
        train_load = DataLoader(train_data, batch_size=args.batch_size)
        test_load = DataLoader(test_data, batch_size=args.batch_size)

        # test_loader = get_mnist_test_loader(dir_name='./RawModels/MNIST/',
        #                                 batch_size=MNIST_Training_Parameters['batch_size'])
        # train_loader, _,train_data = get_mnist_train_validate_loader(dir_name='./RawModels/MNIST/',
        #                                               batch_size=MNIST_Training_Parameters['batch_size'])
        # # train_data = MNIST(root='./RawModels/MNIST/',train=True,transform=transforms.ToTensor())
        raw_model = MNISTConvNet().to(device)
        raw_model.load(rawModel_path,device)
        raw_model = PyTorchClassifier(raw_model,loss=None,input_shape=(1,28,28),nb_classes=10)
        nb_layer = 6
    if args.dataset == 'SVHN':
        from RawModels.SVHNConv import SVHNConvNet
        rawModel_path = 'RawModels/SVHN/model/SVHN_raw.pt'
        train_data = SVHN(root='./RawModels/SVHN/', split='train', transform=transforms.ToTensor())
        test_data = SVHN(root='./RawModels/SVHN/', split='test', transform=transforms.ToTensor())
        X_train = train_data.data
        Y_train = train_data.labels
        X_test = test_data.data
        y_test = test_data.labels
        train_load = DataLoader(train_data, batch_size=args.batch_size)
        test_load = DataLoader(test_data, batch_size=args.batch_size)
        print(Y_train.shape)
        raw_model = SVHNConvNet().to(device)
        raw_model.load(rawModel_path,device)
        raw_model = PyTorchClassifier(raw_model,loss=None,input_shape=(3,32,32),nb_classes=10)
        nb_layer =6
    if args.dataset == 'CIFAR10':
        from RawModels.ResNet import resnet20_cifar, CIFAR10_Training_Parameters, None_feature, get_feature
        rawModel_path = 'RawModels/CIFAR10/model/CIFAR10_raw.pt'
        train_data = CIFAR10(root='./RawModels/CIFAR10/', train=True, transform=transforms.ToTensor())
        test_data = CIFAR10(root='./RawModels/CIFAR10/', train=False, transform=transforms.ToTensor())
        X_train = train_data.data
        Y_train = train_data.targets
        X_test = test_data.data
        Y_test = test_data.targets
        train_load = DataLoader(train_data, batch_size=args.batch_size)

        print(type(np.asarray(Y_train)))
        print(torch.from_numpy(np.asarray(Y_train)).size(0))
        test_load = DataLoader(test_data, batch_size=args.batch_size)
        raw_model = resnet20_cifar().to(device)
        raw_model.load(rawModel_path,device)
        raw_model = PyTorchClassifier(raw_model,loss=None,input_shape=(3,32,32),nb_classes=10)
        nb_layer = 9
    if args.dataset == 'IMAGENET10':
        from RawModels.Resnet_for_imagenet import resnet34_imagenet, loadData

        rawModel_path = 'RawModels/IMAGENET10/model/IMAGENET10_raw.pt'
        train_loader, test_loader = loadData(path='./RawModels/IMAGENET10/dataset/',
                                             batch_size=32)

        raw_model = resnet34_imagenet()
        raw_model.load_state_dict(torch.load(rawModel_path))
        raw_model.to(device)
        raw_model = PyTorchClassifier(raw_model, loss=None, input_shape=(3, 224, 224), nb_classes=10)
        nb_layer =9
    if not os.path.exists(rawModel_path):
        print('please train model first!')

    # Load the dataset
    # X_train, _, X_test, Y_test = model_class.x_train, model_class.y_train, model_class.x_test, model_class.y_test

    # -----------------------------------------------#
    #         Generate layers data Normal           #
    #       Load it if it is already generated      #
    # -----------------------------------------------#
    adv_load = adv_loader(args.attack, args.dataset, args.batch_size)
    layers_data_baseline_path = 'nic_result/{}_layers_baseline.npy'.format(args.dataset)
    layers_data_train_path = 'nic_result/{}_layers_trian.npy'.format(args.dataset)
    layers_data_test_path = 'nic_result/{}_layers_test.npy'.format(args.dataset)
    layers_data_adv_path = 'nic_result/{}_layers_{}.npy'.format(args.dataset, args.attack)
    if not os.path.isfile(layers_data_test_path):
        test_features,test_label = get_layers_feature(raw_model, test_load, nb_layer, device, 100)
        np.save(layers_data_test_path, [test_features,test_label])
    else:
        test_features,test_label = np.load(layers_data_test_path, allow_pickle=True)
    if not os.path.isfile(layers_data_adv_path):
        adv_features,adv_label = get_layers_feature(raw_model, adv_load, nb_layer, device, rand=0)
        np.save(layers_data_adv_path, [adv_features,adv_label ])
    else:
        adv_features,adv_label = np.load(layers_data_adv_path, allow_pickle=True)
    if not os.path.isfile(layers_data_train_path):
        train_features,train_label = get_layers_feature(raw_model, train_load, nb_layer, device=device, rand=0)
        np.save(layers_data_train_path, [train_features,train_label])
    else:
        train_features,train_label = np.load(layers_data_train_path, allow_pickle=True)
    # train_features,train_label = get_layers_feature(raw_model,train_load,nb_layer,device,0)
    # test_features,test_label = get_layers_feature(raw_model,test_load,nb_layer,device,1000)
    #
    # adv_features,adv_label = get_layers_feature(raw_model,adv_load,nb_layer,device,0)
    nic_layers_dir='nic_result/layers/'
    n_layers = len(train_features)
    projector = PCA(n_components=5000)
    # for train
    for l_indx in range(0, n_layers):
        layer_data_path = '{}{}_{}_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
        # if not os.path.isfile(layer_data_path):
            # curr_model = Model(inputs=model.get_input_at(0), outputs=model.get_layer(index=l_indx).output)
            # l_out = curr_model.predict(X_train)
            # l_out = l_out.reshape((X_train.shape[0], -1))
        l_out = train_features[l_indx]
        print(l_out.shape)
        if l_out.shape[1] > 5000:
            start = time.time()
            reduced_activations = projector.fit_transform(l_out)
            end = time.time()
            print('all time is: {}'.format(end-start))
            np.save(layer_data_path, reduced_activations)
        else:
            np.save(layer_data_path, l_out)
        print(layer_data_path)

        # for test
        # for l_indx in range(start_indx, n_layers):
        layer_data_path = '{}{}_{}_test.npy'.format(nic_layers_dir, args.dataset, l_indx)
        # if args.dataset == 'svhn':
        #     X_test = X_test[:10000]
        if not os.path.isfile(layer_data_path):
            # curr_model = Model(inputs=model.get_input_at(0), outputs=model.get_layer(index=l_indx).output)
            # l_out = curr_model.predict(X_test)
            # l_out = l_out.reshape((X_test.shape[0], -1))
            l_out = test_features[l_indx]
            if l_out.shape[1] > 5000:
                # projector = PCA(n_components=5000)
                reduced_activations = projector.transform(l_out)
                np.save(layer_data_path, reduced_activations)
            else:
                np.save(layer_data_path, l_out)
            print(layer_data_path)
        layer_data_path = '{}{}_{}_{}.npy'.format(nic_layers_dir, args.dataset, l_indx,args.attack)
        # if args.dataset == 'svhn':
        #     X_test = X_test[:10000]
        if not os.path.isfile(layer_data_path):
            # curr_model = Model(inputs=model.get_input_at(0), outputs=model.get_layer(index=l_indx).output)
            # l_out = curr_model.predict(X_test)
            # l_out = l_out.reshape((X_test.shape[0], -1))
            l_out = adv_features[l_indx]
            if l_out.shape[1] > 5000:
                # projector = PCA(n_components=5000)
                reduced_activations = projector.transform(l_out)
                np.save(layer_data_path, reduced_activations)
            else:
                np.save(layer_data_path, l_out)
            print(layer_data_path)
        # -----------------------------------------------#
        #        Generate layers data Adv attack        #
        #       Load it if it is already generated      #
        # -----------------------------------------------#
        # for attack in ATTACKS:
        #     X_adv = np.load('%s%s_%s.npy' % (adv_data_dir, args.dataset, attack))
        #     if args.dataset == 'svhn':
        #         X_adv = X_adv[:10000]
        #     # for l_indx in range(start_indx, n_layers):
        #     layer_data_path = '{}{}_{}_{}.npy'.format(nic_layers_dir, args.dataset, l_indx, attack)
        #     if not os.path.isfile(layer_data_path):
        #         curr_model = Model(inputs=model.get_input_at(0), outputs=model.get_layer(index=l_indx).output)
        #         l_out = curr_model.predict(X_adv)
        #         l_out = l_out.reshape((X_adv.shape[0], -1))
        #         if l_out.shape[1] > 5000:
        #             # projector = PCA(n_components=5000)
        #             reduced_activations = projector.transform(l_out)
        #             np.save(layer_data_path, reduced_activations)
        #         else:
        #             np.save(layer_data_path, l_out)
        #         print(layer_data_path)

        # -----------------------------------------------#
        #        Generate layers data gray attack       #
        #       Load it if it is already generated      #
        # -----------------------------------------------#
        # for attack in ATTACK_GRAY[DATASETS.index(args.dataset)]:
        #     if not (attack == 'df' and args.dataset == 'tiny'):
        #         X_adv = np.load('%s%s_%s.npy' % (adv_data_gray_dir, args.dataset, attack))
        #         if args.dataset == 'svhn':
        #             X_adv = X_adv[:10000]
        #         # for l_indx in range(start_indx, n_layers):
        #         layer_data_path = '{}{}_{}_{}.npy'.format(nic_layers_gray_dir, args.dataset, l_indx, attack)
        #         if not os.path.isfile(layer_data_path):
        #             curr_model = Model(inputs=model.get_input_at(0), outputs=model.get_layer(index=l_indx).output)
        #             l_out = curr_model.predict(X_adv)
        #             l_out = l_out.reshape((X_adv.shape[0], -1))
        #             if l_out.shape[1] > 5000:
        #                 # projector = PCA(n_components=5000)
        #                 reduced_activations = projector.transform(l_out)
        #                 np.save(layer_data_path, reduced_activations)
        #             else:
        #                 np.save(layer_data_path, l_out)
        #             print(layer_data_path)

    # -----------------------------------------------#
    #                  Train VIs                    #
    # -----------------------------------------------#
    min_features = 5000
    for l_indx in range(0, n_layers):
        layer_data_path = '{}{}_{}_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
        model_path = '{}{}_{}_pi.model'.format(nic_layers_dir, args.dataset, l_indx)
        full_connect_model_path='{}{}_{}.pth'.format(nic_layers_dir,args.dataset,l_indx)
        pi_predict_normal_path = '{}{}_{}_pi_predict_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
        pi_decision_normal_path = '{}{}_{}_pi_decision_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
        if not os.path.isfile(model_path):
            if os.path.isfile(layer_data_path):
                layer_data = np.load(layer_data_path)
                n_features = np.min([min_features, layer_data.shape[1]])
                layer_data = layer_data[:, :n_features]
                clf = OneClassSVM(nu=0.01, kernel="rbf", gamma=1, verbose=True)
                st = time.time()
                clf.fit(layer_data)
                predict_result = clf.predict(layer_data)
                decision_result = clf.decision_function(layer_data)
                # Saving
                s = pickle.dumps(clf)
                f= open(model_path,'wb+')
                f.write(s)
                f.close()
                # clf.save_to_file(model_path)
                np.save(pi_predict_normal_path, predict_result)
                np.save(pi_decision_normal_path, decision_result)
                # current_model=dense(layer_data.shape[1],args)
                et = time.time()
                t = round((et - st) / 60, 2)
                print('Training PI on {}, layer {} is completed on {} min(s).'.format(args.dataset, l_indx, t))
        if not os.path.isfile(full_connect_model_path):
            if os.path.isfile(layer_data_path):
                st = time.time()
                layer_data = np.load(layer_data_path)
                n_features = np.min([min_features, layer_data.shape[1]])
                layer_data = layer_data[:, :n_features]
                current_model = dense(layer_data.shape[1], args)
                train(current_model, layer_data, train_label, args)
                torch.save(current_model, full_connect_model_path)
                et = time.time()
                t = round((et - st) / 60, 2)
                print('Training full_connect_model on {}, layer {} is completed on {} min(s).'.format(args.dataset, l_indx, t))

    # -----------------------------------------------#
    #                  Train PIs                    #
    # -----------------------------------------------#
    for l_indx in range(0, n_layers - 1):
        layer_data_path_current = '{}{}_{}_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
        layer_data_path_next = '{}{}_{}_normal.npy'.format(nic_layers_dir, args.dataset, l_indx + 1)
        model_path = '{}{}_{}_vi.model'.format(nic_layers_dir, args.dataset, l_indx)
        full_connect_model_path = '{}{}_{}.pth'.format(nic_layers_dir, args.dataset, l_indx)
        full_connect_model_path_next = '{}{}_{}.pth'.format(nic_layers_dir, args.dataset, l_indx+1)
        vi_train_path = '{}{}_{}_vi_train.npy'.format(nic_layers_dir, args.dataset, l_indx)
        vi_predict_normal_path = '{}{}_{}_vi_predict_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
        vi_decision_normal_path = '{}{}_{}_vi_decision_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
        if not os.path.isfile(model_path):
            if os.path.isfile(layer_data_path_current) & os.path.isfile(layer_data_path_next):
                layer_data_current = np.load(layer_data_path_current)
                layer_data_next = np.load(layer_data_path_next)
                n_features_current = np.min([min_features, layer_data_current.shape[1]])
                layer_data_current = layer_data_current[:, :n_features_current]
                n_features_next = np.min([min_features, layer_data_next.shape[1]])
                layer_data_next = layer_data_next[:, :n_features_next]

                model_current = dense(layer_data_current.shape[1],args)  # 不需要训练？
                if os.path.exists(full_connect_model_path):
                    model_current = torch.load(full_connect_model_path)
                # train(model_current,layer_data_current,Y_train,args)
                vi_current = model_current(torch.from_numpy(layer_data_current).to(device))
                model_next=dense(layer_data_next.shape[1],args)
                if os.path.exists(full_connect_model_path_next):
                    model_next = torch.load(full_connect_model_path_next)

                vi_next = model_next(torch.from_numpy(layer_data_next).to(device))
                vi_train = np.concatenate((vi_current.cpu().detach().numpy(), vi_next.cpu().detach().numpy()), axis=1)
                np.save(vi_train_path, vi_train)

                clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='scale', verbose=True)
                st = time.time()
                clf.fit(vi_train)
                predict_result = clf.predict(vi_train)
                decision_result = clf.decision_function(vi_train)
                # Saving
                # clf.save_to_file(model_path)
                s = pickle.dumps(clf)
                f = open(model_path, "wb+")
                f.write(s)
                f.close()
                np.save(vi_predict_normal_path, predict_result)
                np.save(vi_decision_normal_path, decision_result)
                et = time.time()
                t = round((et - st) / 60, 2)
                print('Training VI on {}, layer {} is completed on {} min(s).'.format(args.dataset, l_indx, t))

    # -----------------------------------------------#
    #                  Train NIC                    #
    # Train detector -- if already trained, load it #
    # -----------------------------------------------#
    nic_results_dir = 'nic_result/'
    nic_model_path = '{}{}_nic.model'.format(nic_results_dir, args.dataset)
    nic_train_path = '{}{}_nic_train.npy'.format(nic_layers_dir, args.dataset)
    nic_predict_normal_path = '{}{}_nic_predict_normal.npy'.format(nic_layers_dir, args.dataset)
    nic_decision_normal_path = '{}{}_nic_decision_normal.npy'.format(nic_layers_dir, args.dataset)
    if not os.path.isfile(nic_train_path):
        # collect pis
        pis = np.array([])
        for l_indx in range(0, n_layers):
            pi_decision_normal_path = '{}{}_{}_pi_decision_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
            if os.path.isfile(pi_decision_normal_path):
                pi = np.load(pi_decision_normal_path)
                pi = pi[:, np.newaxis]
                if pis.size == 0:
                    pis = pi
                else:
                    pis = np.concatenate((pis, pi), axis=1)
        # collect pis
        vis = np.array([])
        for l_indx in range(0, n_layers - 1):
            vi_decision_normal_path = '{}{}_{}_vi_decision_normal.npy'.format(nic_layers_dir, args.dataset, l_indx)
            if os.path.isfile(vi_decision_normal_path):
                vi = np.load(vi_decision_normal_path).reshape(-1, 1)
                # vi = vi[:, np.newaxis]
                if vis.size == 0:
                    vis = vi
                else:
                    vis = np.concatenate((vis, vi), axis=1)
        # nic train data
        print(pis.shape)
        print(vis.shape)
        nic_train = np.concatenate((pis, vis), axis=1)
        np.save(nic_train_path, nic_train)
    else:
        nic_train = np.load(nic_train_path)

    train_inds_path = '{}{}_train_inds.npy'.format(nic_results_dir, args.dataset)
    if not os.path.isfile(train_inds_path):
        train_inds = random.sample(range(len(nic_train)), np.int(0.8 * len(nic_train)))
        np.save(train_inds_path, train_inds)
    else:
        train_inds = np.load(train_inds_path)
    test_inds = np.asarray(list(set(range(len(nic_train))) - set(train_inds)))

    if not os.path.isfile(nic_model_path):
        # train nic
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='scale', verbose=True)
        st = time.time()
        clf.fit(nic_train[train_inds])
        predict_result = clf.predict(nic_train[train_inds])
        decision_result = clf.decision_function(nic_train[train_inds])
        # Saving
        # clf.save_to_file(nic_model_path)
        s = pickle.dumps(clf)
        f = open(nic_model_path, "wb+")
        f.write(s)
        f.close()
        np.save(nic_predict_normal_path, predict_result)
        np.save(nic_decision_normal_path, decision_result)
        et = time.time()
        t = round((et - st) / 60, 2)
        print('Training NIC on {} is completed on {} min(s).'.format(args.dataset, t))

    # Refine the normal, noisy and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    # preds_test = model.predict(X_test)
    # if args.dataset == 'svhn':
    #     Y_test = Y_test[:10000]
    # inds_correct = np.where(preds_test.argmax(axis=1) == Y_test.argmax(axis=1))[0]
    # print("Number of correctly predict images: %s" % (len(inds_correct)))
    # X_test = X_test[inds_correct]
    # Y_test = Y_test[inds_correct]
    # print("X_test: ", X_test.shape)

    #-----------------------------------------------#
    #              Prepare Test Data                #
    #-----------------------------------------------#
    #get pi decision for each layer
    #a-load pi_model_normal of the layer, b- load/save the decisions of adv
    # pis = np.array([])
    # for l_indx in range(0, n_layers):
    #     layer_test_path = '{}{}_{}_test.npy'.format(nic_layers_dir, args.dataset, l_indx)
    #     model_path = '{}{}_{}_pi.model'.format(nic_layers_dir, args.dataset, l_indx)
    #     pi_decision_test_path = '{}{}_{}_pi_decision_test.npy'.format(nic_layers_dir, args.dataset, l_indx)
    #     if not os.path.isfile(pi_decision_test_path):
    #         if os.path.isfile(layer_test_path) & os.path.isfile(model_path):
    #             layer_data = np.load(layer_test_path)
    #             n_features = np.min([min_features, layer_data.shape[1]])
    #             layer_data = layer_data[:,:n_features]
    #             clf = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1, verbose=True)
    #             clf.load_from_file(model_path)
    #             decision_result = clf.decision_function(layer_data)
    #             np.save(pi_decision_test_path, decision_result)
    #     else:
    #         decision_result = np.load(pi_decision_test_path)
    #
    #     if pis.size == 0:
    #         pis = decision_result
    #     else:
    #         pis = np.concatenate((pis, decision_result), axis=1)
    #
    # #get vi decision for each layer
    # #a-load vi_model_normal of the layer, b- load/save the decisions of adv
    # vis = np.array([])
    # for l_indx in range(0, n_layers-1):
    #     layer_test_path_current = '{}{}_{}_test.npy'.format(nic_layers_dir, args.dataset, l_indx)
    #     layer_test_path_next = '{}{}_{}_test.npy'.format(nic_layers_dir, args.dataset, l_indx+1)
    #     model_path = '{}{}_{}_vi.model'.format(nic_layers_dir, args.dataset, l_indx)
    #     vi_decision_test_path = '{}{}_{}_vi_decision_test.npy'.format(nic_layers_dir, args.dataset, l_indx)
    #     if not os.path.isfile(vi_decision_test_path):
    #         if os.path.isfile(layer_test_path_current) & os.path.isfile(layer_test_path_next) & os.path.isfile(model_path):
    #             layer_data_current = np.load(layer_test_path_current)
    #             n_features = np.min([min_features, layer_data_current.shape[1]])
    #             layer_data_current = layer_data_current[:,:n_features]
    #             layer_data_next = np.load(layer_test_path_next)
    #             n_features = np.min([min_features, layer_data_next.shape[1]])
    #             layer_data_next = layer_data_next[:,:n_features]
    #             model_current = dense(layer_data_current.shape)  # 不需要训练？
    #             if os.path.exists(full_connect_model_path):
    #                 model_current = torch.load(full_connect_model_path)
    #             # train(model_current,layer_data_current,Y_train,args)
    #             vi_current = model_current(layer_data_current)
    #             if os.path.exists(full_connect_model_path_next):
    #                 model_next = torch.load(full_connect_model_path_next)
    #             vi_next = model_next(layer_data_next)
    #             vi_test_train = np.concatenate((vi_current, vi_next), axis=1)
    #
    #             clf = pickle.load(open(model_path, 'rb'))
    #             decision_result = clf.decision_function(vi_test_train).reshape(-1, 1)
    #             np.save(vi_decision_test_path, decision_result)
    #     else:
    #         decision_result = np.load(vi_decision_test_path)
    #
    #     if vis.size == 0:
    #         vis = decision_result
    #     else:
    #         vis = np.concatenate((vis, decision_result), axis=1)
    #
    # nic_test = np.concatenate((pis, vis), axis=1)
    nic_test = nic_train[test_inds]
    # nic_test_copy=nic_test
    # nic_test = nic_test[inds_correct]

    # -----------------------------------------------#
    #                 Evaluate NIC                  #
    # -----------------------------------------------#
    ## Evaluate detector -- on adversarial attack
    # Y_test_copy=Y_test
    # X_test_copy=X_test
    # for attack in ATTACKS:
    #     # Y_test=Y_test_copy
    #     # X_test=X_test_copy
    #     # nic_test=nic_test_copy
    #     # nic_test = nic_test[inds_correct]
    #     results_all = []

        # get pi decision for each layer
        # a-load pi_model_normal of the layer, b- load/save the decisions of adv
    pis = np.array([])
    for l_indx in range(0, n_layers):
        layer_adv_path = '{}{}_{}_{}.npy'.format(nic_layers_dir, args.dataset, l_indx, args.attack)
        model_path = '{}{}_{}_pi.model'.format(nic_layers_dir, args.dataset, l_indx)
        pi_decision_adv_path = '{}{}_{}_pi_decision_{}.npy'.format(nic_layers_dir, args.dataset, l_indx, args.attack)
        if not os.path.isfile(pi_decision_adv_path):
            if os.path.isfile(layer_adv_path) & os.path.isfile(model_path):
                layer_data = np.load(layer_adv_path)
                n_features = np.min([min_features, layer_data.shape[1]])
                layer_data = layer_data[:, :n_features]
                clf = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1, verbose=True)
                clf = pickle.load(open(model_path, 'rb'))
                # clf.load_from_file(model_path)
                decision_result = clf.decision_function(layer_data)
                np.save(pi_decision_adv_path, decision_result)
        else:
            decision_result = np.load(pi_decision_adv_path)

        if pis.size == 0:
            pis = decision_result[:,np.newaxis]
        else:
            pis = np.concatenate((pis, decision_result[:,np.newaxis]), axis=1)

    # get vi decision for each layer
    # a-load vi_model_normal of the layer, b- load/save the decisions of adv
    vis = np.array([])
    for l_indx in range(0, n_layers - 1):
        layer_adv_path_current = '{}{}_{}_{}.npy'.format(nic_layers_dir, args.dataset, l_indx, args.attack)
        layer_adv_path_next = '{}{}_{}_{}.npy'.format(nic_layers_dir, args.dataset, l_indx + 1, args.attack)
        model_path = '{}{}_{}_vi.model'.format(nic_layers_dir, args.dataset, l_indx)
        vi_decision_adv_path = '{}{}_{}_vi_decision_{}.npy'.format(nic_layers_dir, args.dataset, l_indx, args.attack)
        full_connect_model_path = '{}{}_{}.pth'.format(nic_layers_dir, args.dataset, l_indx)
        full_connect_model_path_next = '{}{}_{}.pth'.format(nic_layers_dir, args.dataset, l_indx+1)
        if not os.path.isfile(vi_decision_adv_path):
            if os.path.isfile(layer_adv_path_current) & os.path.isfile(layer_adv_path_next) & os.path.isfile(
                    model_path):
                layer_data_current = np.load(layer_adv_path_current)
                n_features = np.min([min_features, layer_data_current.shape[1]])
                layer_data_current = layer_data_current[:, :n_features]
                layer_data_next = np.load(layer_adv_path_next)
                n_features = np.min([min_features, layer_data_next.shape[1]])
                layer_data_next = layer_data_next[:, :n_features]
                # model_current = dense(layer_data_current.shape[1],args)
                # vi_current = model_current.predict(layer_data_current)
                # model_next = dense(layer_data_next.shape[1],args)
                model_current = dense(layer_data_current.shape[1], args)  # 不需要训练？
                if os.path.exists(full_connect_model_path):
                    model_current = torch.load(full_connect_model_path)
                # train(model_current,layer_data_current,Y_train,args)
                vi_current = model_current(torch.from_numpy(layer_data_current).to(device))
                model_next = dense(layer_data_next.shape[1], args)
                if os.path.exists(full_connect_model_path_next):
                    model_next = torch.load(full_connect_model_path_next)
                # vi_next = model_next.predict(layer_data_next)
                vi_next = model_next(torch.from_numpy(layer_data_next).to(device))
                vi_adv_train = np.concatenate((vi_current.cpu().detach().numpy(), vi_next.cpu().detach().numpy()), axis=1)

                clf = pickle.load(open(model_path, 'rb'))
                decision_result = clf.decision_function(vi_adv_train).reshape(-1, 1)
                np.save(vi_decision_adv_path, decision_result)
        else:
            decision_result = np.load(vi_decision_adv_path)

        if vis.size == 0:
            vis = decision_result
        else:
            vis = np.concatenate((vis, decision_result), axis=1)

    nic_adv = np.concatenate((pis, vis), axis=1)

    # Prepare data
    # Load adversarial samples
    # X_test_adv = np.load('%s%s_%s.npy' % (adv_data_dir, args.dataset, attack))
    # if attack=='df' and args.dataset=='tiny':
    #     Y_test=model_class.y_test[0:2700]
    #     X_test=model_class.x_test[0:2700]
    #     # nic_test=nic_test_copy[0:2700]
    #     X_test_adv = X_test_adv[0:2700]
    #     cwi_inds = inds_correct[inds_correct<2700]
    #     Y_test = Y_test[cwi_inds]
    #     X_test = X_test[cwi_inds]
    #     nic_test = nic_test[cwi_inds]
    #     nic_adv = nic_adv[cwi_inds]
    #     X_test_adv = X_test_adv[cwi_inds]
    # else:
    # nic_adv = nic_adv[inds_correct]
    # X_test_adv = X_test_adv[inds_correct]
    #
    # pred_adv = model.predict(X_test_adv)
    # loss, acc_suc = model.evaluate(X_test_adv, Y_test, verbose=0)
    # inds_success = np.where(pred_adv.argmax(axis=1) != Y_test.argmax(axis=1))[0]
    # inds_fail = np.where(pred_adv.argmax(axis=1) == Y_test.argmax(axis=1))[0]
    # nic_adv_success = nic_adv[inds_success]
    # nic_adv_fail = nic_adv[inds_fail]

    # prepare X and Y for detectors
    X_all = np.concatenate([nic_test, nic_adv])
    Y_all = np.concatenate([np.ones(len(nic_test), dtype=bool), np.zeros(len(nic_adv), dtype=bool)])
    # X_success = np.concatenate([nic_test, nic_adv_success])
    # Y_success = np.concatenate([np.zeros(len(nic_test), dtype=bool), np.ones(len(inds_success), dtype=bool)])
    # X_fail = np.concatenate([nic_test, nic_adv_fail])
    # Y_fail = np.concatenate([np.zeros(len(nic_test), dtype=bool), np.ones(len(inds_fail), dtype=bool)])

    # --- load nic detector
    clf = pickle.load(open(nic_model_path, 'rb'))

    # For Y_all
    Y_all_pred = clf.predict(X_all)
    Y_all_pred = process(Y_all_pred)
    Y_all_pred_score = clf.decision_function(X_all)

    acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all = evalulate_detection_test(Y_all, Y_all_pred)
    fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, Y_all_pred_score)
    roc_auc_all = auc(fprs_all, tprs_all)
    print("AUC: {:.4f}% ACC: {:.4f}% TPR: {:.4f}% TFR: {:.4f}%".format(100 * roc_auc_all,100*acc_all,100*tpr_all,100*fpr_all))

if __name__ == "__main__":

    main()
