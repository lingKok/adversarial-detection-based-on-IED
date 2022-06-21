import pickle

import torch
import os
import argparse
import numpy as np
import random

from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.svm import SVC

from RawModels.Utils.TrainTest import testing_evaluation, predict
from RawModels.Utils.dataset import get_mnist_test_loader, get_mnist_train_validate_loader,get_svhn_train_validate_loader,get_svhn_test_loader\
    ,get_cifar10_train_validate_loader,get_cifar10_test_loader
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from torchvision.datasets import MNIST,CIFAR10,SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# def get_layers_from_standard_dataload(model,dataload,device):
#     feature = {}
#     model.train()
#     for data, target in dataload:
#         data = data.to(device)
#         target = target.to(device)
#         model(data)
#         tem = model.middle.copy() # {0:(n*d1),1:(n*d1).....}
#         for i in range(len(tem)):
#             # print(tem[i])
#             if feature.get(i,None) is None:
#                 feature[i] = tem[i].cpu().detach().numpy()
#
#             else:
#                 feature[i] = np.concatenate((feature[i], tem[i].cpu().detach().numpy()))
#         del tem
#         torch.cuda.empty_cache()
#     return feature

def get_layers_feature(model,data_loader,nb_layer,device,rand=0):

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
                layers_feature[i]=np.asarray(tem[cor_ind],dtype='float16')


            else:
                tem = model.get_activations(x, i).reshape(len(x), -1)
                layers_feature[i] = np.concatenate((layers_feature[i],np.asarray(tem[cor_ind],dtype='float16')
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

# def get_layers_from_dataload(model,dataload,device):
#     feature={}
#     for data in dataload:
#         data = data.to(device)
#         # target = target.to(device)
#         model(data)
#         tem = model.middle.copy()  # {0:(n*d1),1:(n*d1).....}
#         for i in range(len(tem)):
#             if feature is not None:
#                 feature[i] = np.concatenate((feature[i], tem[i]))
#             else:
#                 feature[i] = tem[i]
#     return feature

# def get_layer_from_dataload(model,dataload,device):
#     feature = {}
#     for data in dataload:
#         data = data.to(device)
#         # target = target.to(device)
#         model(data)
#         tem = model.mid_layer.copy()  # {0:(n*d1),1:(n*d1).....}
#
#         if feature is not None:
#             feature[0] = np.concatenate((feature[0], tem[0]))
#         else:
#             feature[0] = tem[0]
#     return feature


def get_layer_from_standard_dataload(model, dataload, device):
    feature = {}
    model.train()
    for data,target in dataload:
        data = data.to(device)
        # target = target.to(device)
        model(data)
        tem = model.mid_layer.copy()  # {0:(n*d1),1:(n*d1).....}

        if feature is not None:
            feature[0] = np.concatenate((feature[0], tem[0]))
        else:
            feature[0] = tem[0]
    return feature
def train_dnr(args,model,dataload,nb_layer,device):
    dnr_results_dir= 'dnr_results/'+args.dataset+'/dnr'
    features,labels = get_layers_feature(model,dataload,nb_layer,device)
    lengh = len(features)
    clf_output=np.array([])
    for i in range(1,4):

        feature=features[lengh-i]
        clf =SVC(probability=True,random_state=55)
        clf.fit(feature,labels)
        #save classifiers
        s = pickle.dumps(clf)
        f = open('{}{}'.format(dnr_results_dir,'_'+str(i)+'.model'),'wb+')
        f.write(s)
        f.close()
        print('{} classifier training is finished'.format(str(i+1)))
        if clf_output.size==0:
            clf_output=clf.predict_proba(feature)
        else:
            clf_output=np.concatenate((clf_output,clf.predict_proba(feature)),axis=1)
    clf =SVC(probability=True,random_state=55)
    clf.fit(clf_output,labels)
    s=pickle.dumps(clf)
    f=open('{}{}'.format(dnr_results_dir,'.model'),'wb+')
    f.write(s)
    f.close()

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
def get_dnr_feature(args,model,train_load,test_load,adv_load,nb_layer,device):
    dnr_results = 'dnr_results/'+args.dataset+'/dnr.model'
    if not os.path.exists(dnr_results):
        os.makedirs('/dnr_results/{}/'.format(args.dataset),exist_ok=True)
        train_dnr(args,model,train_load,nb_layer,device)
    clfs=[]
    for i in range(1,4):
        clfs.append(pickle.load(open(dnr_results[:-6]+'_'+str(i)+'.model','rb')))
    clfs.append(pickle.load(open(dnr_results,'rb')))
    print('Classifiers loading ...')
    test_features,labels=get_layers_feature(model,test_load,nb_layer,device)
    lengh = len(test_features)
    clf_output=np.array([])
    for i in range(1,4):
        test_feature = test_features[lengh-i][0:1000]
        clf= clfs[i-1]
        if clf_output.size==0:

            clf_output=clf.predict_proba(test_feature)
        else:
            clf_output= np.concatenate((clf_output,clf.predict_proba(test_feature)),axis=1)
    clf= clfs[-1]
    x_test_pred=clf.predict(clf_output)
    x_test_score=np.max(clf.predict_proba(clf_output),axis=1)

    thr = np.percentile(x_test_score,10)
    reject_inds_clear=[i for i,v in enumerate(x_test_score) if np.max(x_test_score[i])<=thr]
    adv_features ,adv_labels= get_layers_feature(model,adv_load,nb_layer,device)
    lengh=len(adv_features)
    clf_output_adv =np.array([])
    for i in range(1,4):
        adv_feature = adv_features[lengh-i]
        clf=clfs[i-1]
        adv_feature[np.isnan(adv_feature)]=0
        if clf_output_adv.size==0:
            clf_output_adv = clf.predict_proba(adv_feature)

        else:
            clf_output_adv = np.concatenate((clf_output_adv,clf.predict_proba(adv_feature)),axis=1)

    clf=clfs[-1]
    x_adv_pred = clf.predict(clf_output_adv)
    x_adv_score = np.max(clf.predict_proba(clf_output_adv),axis=1)
    reject_inds_adv = [i for i,v in enumerate(x_adv_score)if np.max(x_adv_score[i])<=thr]

    y_clean_pred = np.zeros(len(x_test_pred),dtype=bool)
    y_clean_pred[reject_inds_clear]=True
    y_adv_pred = np.zeros(len(x_adv_pred),dtype= bool)
    y_adv_pred[reject_inds_adv]=True
    y_all=np.concatenate((np.zeros(len(x_test_pred),dtype=int),np.ones(len(x_adv_pred),dtype=int)))
    y_all_pred=np.concatenate((y_clean_pred,y_adv_pred))

    dec = np.ones(len(y_all_pred))
    for i in range(len(dec)):
        if y_all_pred[i]==False:
            dec[i]=-1
    acc, tpr, fpr, tp, ap, fp, an = evalulate_detection_test(y_all, y_all_pred)
    fprs_all, tprs_all, thresholds_all = roc_curve(y_all, y_all_pred)
    roc_auc = auc(fprs_all, tprs_all)

    print("ACC: {:.4f}%, TPR :{:.4f}% FPR : {:.4f}% AUC : {:.4f}%".format(acc*100,tpr*100,fpr*100,roc_auc*100))
    # plot_auc(y_all,y_all_pred,args.attack,args.dataset,'dnr')



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




def adv_loader(args):
    from torch.utils.data import TensorDataset, DataLoader
    advdata_path = './AdversarialExampleDatasets/' + args.attack + '/' + args.dataset + '/' + args.attack + '_AdvExamples.npy'
    Advlabel_path = './AdversarialExampleDatasets/' + args.attack + '/' + args.dataset + '/' + args.attack + '_AdvLabels.npy'
    adv_data = np.load(advdata_path)
    adv_label = np.load(Advlabel_path)
    adv_data = torch.from_numpy(adv_data)
    adv_label = torch.from_numpy(adv_label)
    adv_data = adv_data.to(torch.float32)
    # _, true_label = torch.max(true_label, 1)
    # print(true_label.shape)
    adv_dataset = TensorDataset(adv_data, adv_label)
    adv_loader = DataLoader(adv_dataset, batch_size=args.batch_size)
    return adv_loader


def adv_label(attack_type, dataset_type):
    path = './AdversarialExampleDatasets/' + attack_type + '/' + dataset_type + '/' + attack_type + '_AdvLabels.npy'
    adv_labels = np.load(path)
    adv_labels = torch.from_numpy(adv_labels)
    return adv_labels




def main():
    import argparse
    from art.estimators.classification import PyTorchClassifier
    parse= argparse.ArgumentParser(description='this is an adversarial detection based on Local Cosine Similarity')
    parse.add_argument('--seed', type=int, default=100, help='set the random seed')
    parse.add_argument('--gpu_index', type=str, default='1', help='the index of gpu')
    parse.add_argument('--dataset',default='MNIST',choices=['MNIST','SVHN','CIFAR10','IMAGENET10'])
    parse.add_argument('--attack',default='FGSM',choices=['FGSM','PGD','DEEPFOOL','JSMA','CWL2','CWL0','CWLINF'])
    parse.add_argument('--test_attack', default='FGSM', choices=['FGSM', 'PGD', 'DEEPFOOL', 'JSMA', 'CW2'])
    parse.add_argument('--batch_size',default=64)
    args=parse.parse_args()
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

    if args.dataset=='MNIST':
        from RawModels.MNISTConv import MNISTConvNet
        rawModel_path = 'RawModels/MNIST/model/MNIST_raw.pt'
        train_data=MNIST(root='./RawModels/MNIST/',train=True,transform=transforms.ToTensor())
        test_data= MNIST(root='./RawModels/MNIST/',train=False,transform=transforms.ToTensor())
        train_target=train_data.targets
        train_load=DataLoader(train_data,batch_size=args.batch_size)
        test_load=DataLoader(test_data,batch_size=args.batch_size)


        # test_loader = get_mnist_test_loader(dir_name='./RawModels/MNIST/',
        #                                 batch_size=MNIST_Training_Parameters['batch_size'])
        # train_loader, _,train_data = get_mnist_train_validate_loader(dir_name='./RawModels/MNIST/',
        #                                               batch_size=MNIST_Training_Parameters['batch_size'])
        # # train_data = MNIST(root='./RawModels/MNIST/',train=True,transform=transforms.ToTensor())
        raw_model = MNISTConvNet().to(device)
        raw_model.load(path=rawModel_path, device=device)
        raw_model = PyTorchClassifier(raw_model,loss=None,input_shape=(1,28,28),nb_classes=10)
        nb_layer = 6
    if args.dataset == 'SVHN':
        from RawModels.SVHNConv import SVHNConvNet
        rawModel_path = 'RawModels/SVHN/model/SVHN_raw.pt'
        train_data= SVHN(root='./RawModels/SVHN/',split='train',transform=transforms.ToTensor())
        test_data= SVHN(root='./RawModels/SVHN/',split='test',transform=transforms.ToTensor())
        train_target=train_data.labels
        train_load = DataLoader(train_data, batch_size=args.batch_size)
        test_load = DataLoader(test_data,batch_size=args.batch_size)
        # targets=train_data.labels
        # test_loader = get_svhn_test_loader(dir_name='./RawModels/SVHN/',
        #                                    batch_size=SVHN_Training_Parameters['batch_size'])
        # train_loader,_,train_data = get_svhn_train_validate_loader(dir_name='./RawModels/SVHN/',
        #                                                 batch_size=SVHN_Training_Parameters['batch_size'])
        # train_data = SVHN(root='./RawModels/SVHN/',split='train',transform=transforms.ToTensor())
        raw_model = SVHNConvNet().to(device)
        raw_model.load(path=rawModel_path, device=device)
        raw_model = PyTorchClassifier(raw_model, loss=None, input_shape=(3, 32, 32), nb_classes=10)
        nb_layer = 6
    if args.dataset == 'CIFAR10':
        from RawModels.ResNet import resnet20_cifar, CIFAR10_Training_Parameters, None_feature, get_feature
        rawModel_path = 'RawModels/CIFAR10/model/CIFAR10_raw.pt'
        train_data=CIFAR10(root='./RawModels/CIFAR10/',train=True,transform=transforms.ToTensor())
        test_data = CIFAR10(root='./RawModels/CIFAR10/',train=False,transform=transforms.ToTensor())
        train_target=train_data.targets
        train_load = DataLoader(train_data, batch_size=args.batch_size)
        test_load= DataLoader(test_data,batch_size=args.batch_size)
        # test_loader = get_cifar10_test_loader(dir_name='./RawModels/CIFAR10/',
        #                                       batch_size=CIFAR10_Training_Parameters['batch_size'])
        # train_loader, _ ,train_data=get_cifar10_train_validate_loader(dir_name='./RawModels/CIFAR10/',
        #                                                    batch_size=CIFAR10_Training_Parameters['batch_size'])
        # train_data = CIFAR10(root='./RawModels/CIFAR10/',train=True,transform=transforms.ToTensor())
        raw_model=resnet20_cifar().to(device)
        raw_model.load(path=rawModel_path, device=device)
        raw_model = PyTorchClassifier(raw_model, loss=None, input_shape=(3, 32, 32), nb_classes=10)
        nb_layer = 9
    if args.dataset == 'IMAGENET10':
        from RawModels.Resnet_for_imagenet import resnet34_imagenet, loadData

        rawModel_path = 'RawModels/IMAGENET10/model/IMAGENET10_raw.pt'
        train_load, test_load = loadData(path='./RawModels/IMAGENET10/dataset/',
                                             batch_size=32)

        raw_model = resnet34_imagenet()
        raw_model.load_state_dict(torch.load(rawModel_path))
        raw_model.to(device)
        raw_model = PyTorchClassifier(raw_model, loss=None, input_shape=(3, 224, 224), nb_classes=10)
        nb_layer = 9

    if not os.path.exists(rawModel_path):
        print('please train model first!')


    adv_load = adv_loader(args)
    get_dnr_feature(args,raw_model,train_load,test_load,adv_load,nb_layer,device)


if __name__ == '__main__':

    main()
