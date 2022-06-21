"""
input dataset: test and adv dataset
process the images and extract features(18 verse 1 image)
output auc
"""
# get feature
import os.path


import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_curve, auc
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN, CIFAR10
from MSCN import *
from sklearn.model_selection import train_test_split
from sklearn import svm
import torchvision.datasets as datasets

def loadData(path='./imagenet',batch_size=10,shuffle=True,val_size =0.0):

    #set the path of dataset
    train_path = os.path.join(path + 'train')
    val_path = os.path.join(path + 'val')

    #set transform method, the default img size of imagenet is 224
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(330),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalizer
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # normalizer
    ])

    #import train\val dataset
    train_data = datasets.ImageFolder(root=train_path, transform=train_transform,)
    valid_data = datasets.ImageFolder(root=val_path, transform=valid_transform,)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(val_size*num_train))
    np.random.seed(0)
    np.random.shuffle(indices)
    train_idx,valid_idx = indices[split:],indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)

    # load dataset into Dataloader
    train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)
    #
    print(len(train_data)) #1151273
    print(len(valid_data)) #50000
    return valid_data
def get_dataset(args):
    if args.dataset=='MNIST':
        test = MNIST(root='RawModels/MNIST',train=False)
        test_data = test.data.unsqueeze(-1).numpy()
    if args.dataset == 'SVHN':
        test = SVHN(root='RawModels/SVHN',split='test')
        test_data = test.data.transpose(0,2,3,1)
    if args.dataset == 'CIFAR10':
        test = CIFAR10(root='RawModels/CIFAR10',train=False)
        print(test.data.shape)
        # test_data = test.data.transpose(0,2,3,1)
        test_data =test.data
    if args.dataset == 'IMAGENET10':
        test = loadData('./RawModels/IMAGENET10/dataset/')
        test_data = test.imgs.transpose(0,2,3,1)
    # else:
    #     print('dataset unrealization!')
    # print(test_data.shape)
    return test_data

def get_advdata(args):
    adv_path='AdversarialExampleDatasets/'+args.attack+'/'+args.dataset+'/'+args.attack+'_AdvExamples.npy'
    adv_data=np.load(adv_path)
    print(adv_data.shape)
    adv_data = adv_data.transpose(0,2,3,1)
    return adv_data

def get_nss_feature(x_f_path,X_test):
    # x_f_path='Feature/{}_{}_nss.npy'.format(args.dataset,args.attack)
    # x_test_path='Feature/{}_test_nss.npy'.format(args.dataset)
    # X_test,Y_test=get_dataset(args)
    if (X_test is None) and (not os.path.isfile(x_f_path)):
        print('Firstly generate test attack!')
        exit()

    if not os.path.isfile(x_f_path):
        x_f=np.array([])
        for img in X_test:
            parameters=calculate_brisque_features(img)
            parameters = parameters.reshape((1,-1))
            if x_f.size == 0:
                x_f = parameters
            else:
                x_f = np.concatenate((x_f,parameters),axis=0)
        np.save(x_f_path,x_f)
    else:
        x_f = np.load(x_f_path)
    return x_f

def plot_auc(y_true, y_pre,attack_type,dataset_type,detect_type):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, threshold = roc_curve(y_true, y_pre)
    roc_auc = auc(fpr, tpr)
    path='./ROC/'+dataset_type+'/'+detect_type+'_'+attack_type+'.npy'
    np.save(path,[fpr,tpr,roc_auc])
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr,
             tpr,
             color='darkorange',
             lw=lw,
             label='ROC curve (area = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot(point[0], point[1], marker='o', color='r')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
def evalulate_detection_test(Y_detect_test, Y_detect_pred):
    accuracy = accuracy_score(Y_detect_test, Y_detect_pred, normalize=True, sample_weight=None)
    tpr, fpr, tp, ap, fp, an = get_tpr_fpr(Y_detect_test, Y_detect_pred)
    return accuracy, tpr, fpr, tp, ap, fp, an

def get_tpr_fpr(true_labels, pred_labels):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

    AP = np.sum(true_labels)
    AN = np.sum(1-true_labels)

    tpr = TP/AP if AP>0 else np.nan
    fpr = FP/AN if AN>0 else np.nan

    return tpr, fpr, TP, AP, FP, AN
def main(args):
    X_test = get_dataset(args)

    X_adv = get_advdata(args)
    args.test_attack = args.attack
    attack_list = ['FGSM', 'PGD', 'DEEPFOOL', 'JSMA', 'CWL0', 'CWL2', 'CWLINF']
    test_attack_list = ['FGSM','PGD','DEEPFOOL','JSMA','CWL0','CWL2','CWLINF']
    for attack in attack_list:
        for test_attack in test_attack_list:
            args.attack = attack
            args.test_attack = test_attack
            if args.attack != args.test_attack:

                x_advtest_path = 'nss_result/{}_{}_nss.npy'.format(args.dataset,args.test_attack)
                x_adv_path = 'nss_result/{}_{}_nss.npy'.format(args.dataset, args.attack)
                x_test_path = 'nss_result/{}_test_nss.npy'.format(args.dataset)
                x_f_test = get_nss_feature(x_test_path, X_test)
                x_f_adv = get_nss_feature(x_adv_path, X_adv)
                x_f_advtest = get_nss_feature(x_advtest_path,None)
                y_f_test = np.zeros(len(x_f_test), dtype=np.uint8)
                y_f_adv = np.ones(len(x_f_adv), dtype=np.uint8)
                y_f_advtest = np.ones(len(x_f_advtest), dtype=np.uint8)
                x_f = np.concatenate((x_f_test, x_f_adv))
                x_ftest = np.concatenate((x_f_test, x_f_advtest))
                y_f = np.concatenate((y_f_test, y_f_adv))
                y_ftest = y_f = np.concatenate((y_f_test, y_f_advtest))
                min_ = np.min(x_f, axis=0)
                max_ = np.max(x_f, axis=0)
                x_f = scale_features(x_f, min_, max_)
                min_ = np.min(x_ftest, axis=0)
                max_ = np.max(x_ftest, axis=0)
                x_ftest = scale_features(x_ftest, min_, max_)

                # non_ind = np.isnan(x_f)
                # print(non_ind)
                # x_f = x_f[~non_ind]
                # y_f = y_f[~non_ind]
                if np.any(np.isnan(x_f)):
                    x_f = np.nan_to_num(x_f)
                if np.any(np.isnan(x_ftest)):
                    x_ftest = np.nan_to_num(x_ftest)

                x_train, _, y_train, _ = train_test_split(x_f, y_f, random_state=34, test_size=0.33)
                _, x_test, _, y_test = train_test_split(x_ftest, y_ftest, random_state=34, test_size=0.33)
            else:
                x_adv_path='nss_result/{}_{}_nss.npy'.format(args.dataset,args.attack)
                x_test_path='nss_result/{}_test_nss.npy'.format(args.dataset)
                x_f_test= get_nss_feature(x_test_path,X_test)
                x_f_adv = get_nss_feature(x_adv_path,X_adv)
                y_f_test = np.zeros(len(x_f_test),dtype=np.uint8)
                y_f_adv = np.ones(len(x_f_adv),dtype=np.uint8)
                x_f= np.concatenate((x_f_test,x_f_adv))
                y_f = np.concatenate((y_f_test,y_f_adv))
                min_= np.min(x_f,axis=0)
                max_ =  np.max(x_f,axis=0)
                x_f = scale_features(x_f,min_,max_)
                # non_ind = np.isnan(x_f)
                # print(non_ind)
                # x_f = x_f[~non_ind]
                # y_f = y_f[~non_ind]
                if np.any(np.isnan(x_f)):
                    x_f = np.nan_to_num(x_f)

                x_train, x_test, y_train, y_test =train_test_split(x_f,y_f,random_state=34,test_size=0.33)
            clf = svm.SVC(C=10,kernel='sigmoid',gamma=0.01, probability=True, random_state=0)
            clf.fit(x_train,y_train)
            pred_test=clf.predict(x_test)
            prob_test = clf.predict_proba(x_test)[:,1]
            acc, tpr, fpr, tp, ap, fp, an = evalulate_detection_test(y_test, pred_test)
            fprs_all, tprs_all, thresholds_all = roc_curve(y_test, prob_test)
            roc_auc = auc(fprs_all, tprs_all)

            # print(
            #     "ACC: {:.4f}%, TPR :{:.4f}% FPR : {:.4f}% AUC : {:.4f}%".format(acc * 100, tpr * 100, fpr * 100, roc_auc * 100))
            print('attack type {} test attack type {} auc {}'.format(attack,test_attack, roc_auc))
            # print(x_f_test.shape)






if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='nss detecting')
    parser.add_argument('--dataset',default='MNIST',choices=['MNIST','SVHN','CIFAR10'])
    parser.add_argument('--attack',default='PGD',choices=['FGSM','PGD','JSMA','DEEPFOOL','CWL0','CWL2','CWLINF'])
    parser.add_argument('--test_attack',default='FGSM',choices=['FGSM','PGD','JSMA','DEEPFOOL','CWL0','CWL2','CWLINF'])
    args=parser.parse_args()
    main(args)

