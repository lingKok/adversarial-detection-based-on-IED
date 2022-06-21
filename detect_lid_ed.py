import torch
import os
import argparse
import numpy as np
import random




from RawModels.Utils.TrainTest import testing_evaluation, predict
from RawModels.Utils.dataset import get_mnist_test_loader, get_mnist_train_validate_loader,get_svhn_train_validate_loader,get_svhn_test_loader\
    ,get_cifar10_train_validate_loader,get_cifar10_test_loader
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist


def plot_auc(y_true, y_pre,attack_type,test_attack_type,dataset_type,detect_type):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, threshold = roc_curve(y_true, y_pre)
    roc_auc = auc(fpr, tpr)
    # path='./ROC/'+dataset_type+'/'+detect_type+'_'+attack_type+'.npy'
    # np.save(path,[fpr,tpr,roc_auc])
    # lw = 2
    # plt.figure(figsize=(10, 10))
    # plt.plot(fpr,
    #          tpr,
    #          color='darkorange',
    #          lw=lw,
    #          label='ROC curve (area = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    # # plt.plot(point[0], point[1], marker='o', color='r')
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()
    print('attack type {} ,test attack type {} auc is : {}'.format(attack_type,test_attack_type,roc_auc))


def mle_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)
    print(data.shape)
    print(batch.shape)
    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, :k]
    a = np.apply_along_axis(f, axis=1, arr=a)
    # print('a is :', a.shape)
    return a


def get_lid_by_random(data, adv_data, batch_size, k=10):

    batch_num = int(np.ceil(adv_data.shape[0] / float(batch_size)))
    lid = None
    # lid_adv = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(adv_data), (i_batch + 1) * batch_size)
        # n_feed = end - start
        # lid_batch = np.zeros(shape=(n_feed, 1))
        # lid_batch_adv = np.zeros(shape=(n_feed, 1))
        # if n_feed > k:
        X_act = data

        X_adv_act = adv_data[start:end]
        # print(X_act.shape)
        # print(X_adv_act.shape)
        lid_batch = mle_batch(X_act, X_adv_act, k)
        # lid_batch_adv = mle_batch(X_act, X_adv_act, k)
        if lid is not None:
            lid = np.concatenate((lid, lid_batch), 0)
            # lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            lid = lid_batch
            # lid_adv = lid_batch_adv
    return lid

def get_score_by_lid(dict1, dict2):
    # assert len(dict1) == len(dict2)
    print(dict1[0][0])
    print('####################################')
    print(dict2[0][0])
    print('####################################')
    dim = len(dict1)
    # base_score = np.zeros((len(dict1[0]), dim))
    adv_score = np.zeros((len(dict2[0]), dim))
    for i in range(dim):
        if isinstance(dict1[i],np.ndarray):
            feature1 = dict1[i]
        else:
            feature1 = dict1[i].cpu().detach().numpy()

        # print('feature1 is :', feature1[0])
        if isinstance(dict2[i],np.ndarray):
            feature2 = dict2[i]
        else:
            feature2 = dict2[i].cpu().detach().numpy()
        # print('feature2 is :', feature2[0])

        adv_score[:, i] = get_lid_by_random(feature1, feature2, batch_size=100, k=10)
    return adv_score


def score_point(dup):
    x, kde = dup
    return kde.score_samples(np.reshape(x, (1, -1)))[0]


def adv_loader(type, dataset, batch_size):
    from torch.utils.data import TensorDataset, DataLoader
    advdata_path = './AdversarialExampleDatasets/' + type + '/' + dataset + '/' + type + '_AdvExamples.npy'
    Advlabel_path = './AdversarialExampleDatasets/' + type + '/' + dataset + '/' + type + '_AdvLabels.npy'
    adv_data = np.load(advdata_path)
    adv_label = np.load(Advlabel_path)
    adv_data = torch.from_numpy(adv_data)
    adv_label = torch.from_numpy(adv_label)
    adv_data = adv_data.to(torch.float32)
    # _, true_label = torch.max(true_label, 1)
    # print(true_label.shape)
    adv_dataset = TensorDataset(adv_data, adv_label)
    adv_loader = DataLoader(adv_dataset, batch_size=batch_size)
    return adv_loader


def adv_label(attack_type, dataset_type):
    path = './AdversarialExampleDatasets/' + attack_type + '/' + dataset_type + '/' + attack_type + '_AdvLabels.npy'
    adv_labels = np.load(path)
    adv_labels = torch.from_numpy(adv_labels)
    return adv_labels


def get_trainlabel(train_dataloader):
    label_arr = None
    for data, label in train_dataloader:
        if label_arr is not None:
            label_arr = np.concatenate((label_arr, label), 0)
        else:
            label_arr = label
    return label_arr


def mix_feature(test_score, adv_score):
    adv_label = np.zeros(len(adv_score))
    test_label = np.ones(len(test_score))
    mix_data = np.concatenate((test_score, adv_score), 0)
    mix_label = np.concatenate((test_label, adv_label), 0)
    return mix_data, mix_label


def get_data(dataloader, size):
    data_arr = None
    label_arr = None
    for data, label in dataloader:
        if data_arr is not None:
            data_arr = torch.cat((data_arr, data), 0)
            label_arr = torch.cat((label_arr, label), 0)
        else:
            data_arr = data
            label_arr = label

    return data_arr[0:size], label_arr[0:size]


def get_kde_score(train_feature, train_labels, test_feature, test_labels, adv_feature, adv_labels):
    import pickle
    def score_samples(kdes, samples, preds, n_jobs):
        import multiprocessing as mp
        if n_jobs is not None:
            p = mp.Pool(n_jobs)
        else:
            p = mp.Pool()
        result = np.asarray(
            p.map(score_point, [(x, kdes[preds]) for x, preds in zip(samples, preds)])
        )
        p.close()
        p.join()
        return result
    adv_labels = adv_labels.numpy()
    test_labels = test_labels.numpy()
    class_idx = {}
    for i in range(10):
        class_idx[i] = np.where(train_labels == i)
        # print(class_idx[i][0].shape)
    kdes = {}

    for i in range(10):
        kdes[i] = KernelDensity(bandwidth=0.26, kernel='gaussian').fit(train_feature[class_idx[i]],
                                                                       train_labels[class_idx[i]])

    test_score = score_samples(kdes, test_feature, test_labels, n_jobs=1)
    test_score = test_score.reshape((-1, 1))
    adv_ind=np.isnan(adv_feature)
    adv_feature[adv_ind]=0
    adv_score = score_samples(kdes, adv_feature, adv_labels, n_jobs=1)
    adv_score = adv_score.reshape((-1, 1))

    return test_score, adv_score

def get_layers_from_standard_dataload(model,dataload,device):
    feature = {}
    model.train()
    label = np.array([])
    for data, target in dataload:
        data = data.to(device)
        target = target.to(device)

        output=model(data)
        correct_ind = (torch.argmax(output,1).eq(target))
        # print(correct_ind)
        tem = model.middle.copy() # {0:(n*d1),1:(n*d1).....}
        for i in range(len(tem)):
            # print(tem[i])
            if feature.get(i,None) is None:
                feature[i] = tem[i][correct_ind].cpu().detach().numpy()

            else:
                feature[i] = np.concatenate((feature[i], tem[i][correct_ind].cpu().detach().numpy()))
        if label.shape==0:
            label=target[correct_ind].cpu().numpy()
        else:
            label = np.concatenate((label,target[correct_ind].cpu().numpy()),0)
    return feature,label



def get_baseline(args,model,dataload,device):
    import random
    features,labels = get_layers_from_standard_dataload(model,dataload,device)
    lengh = len(features[0])
    dim = len(features)
    ind_sample = random.sample(range(lengh),1000)
    layers_data_baseline = {}
    for i in range(dim):
        layers_data_baseline[i]=features[i][ind_sample,:]
    labels = labels[ind_sample]
    return layers_data_baseline,labels


def main():
    import argparse
    parse= argparse.ArgumentParser(description='this is an adversarial detection based on Local Cosine Similarity')
    parse.add_argument('--seed', type=int, default=100, help='set the random seed')
    parse.add_argument('--gpu_index', type=str, default='1', help='the index of gpu')
    parse.add_argument('--dataset',default='mnist',choices=['MNIST','SVHN','CIFAR10'])
    parse.add_argument('--type',default='cs',choices=['knn','lcs','cs','kd','lid','iacs','ilacs'])
    parse.add_argument('--attack',default='fgsm',choices=['FGSM','PGD','DEEPFOOL','JSMA','CWL0','CWL2','CWLINF'])
    parse.add_argument('--test_attack', default='fgsm', choices=['FGSM','PGD','DEEPFOOL','JSMA','CWL0','CWL2','CWLINF'])
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
        from RawModels.MNISTConv import MNISTConvNet, MNIST_Training_Parameters, None_feature, get_feature
        rawModel_path = 'RawModels/MNIST/model/MNIST_raw.pt'

        test_loader = get_mnist_test_loader(dir_name='./RawModels/MNIST/',
                                        batch_size=MNIST_Training_Parameters['batch_size'])
        train_loader, _ ,_= get_mnist_train_validate_loader(dir_name='./RawModels/MNIST/',
                                                      batch_size=MNIST_Training_Parameters['batch_size'])
        raw_model = MNISTConvNet().to(device)
        raw_model.load(path=rawModel_path, device=device)
    if args.dataset == 'SVHN':
        from RawModels.SVHNConv import SVHNConvNet, SVHN_Training_Parameters, None_feature, get_feature
        rawModel_path = 'RawModels/SVHN/model/SVHN_raw.pt'
        test_loader = get_svhn_test_loader(dir_name='./RawModels/SVHN/',
                                           batch_size=SVHN_Training_Parameters['batch_size'])
        train_loader,_,_ = get_svhn_train_validate_loader(dir_name='./RawModels/SVHN/',
                                                        batch_size=SVHN_Training_Parameters['batch_size'])
        raw_model = SVHNConvNet().to(device)
        raw_model.load(path=rawModel_path, device=device)
    if args.dataset == 'CIFAR10':
        from RawModels.ResNet import resnet20_cifar, CIFAR10_Training_Parameters, None_feature, get_feature
        rawModel_path = 'RawModels/CIFAR10/model/CIFAR10_raw.pt'
        test_loader = get_cifar10_test_loader(dir_name='./RawModels/CIFAR10/',
                                              batch_size=CIFAR10_Training_Parameters['batch_size'])
        train_loader, _ ,_=get_cifar10_train_validate_loader(dir_name='./RawModels/CIFAR10/',
                                                           batch_size=CIFAR10_Training_Parameters['batch_size'])
        raw_model=resnet20_cifar().to(device)
        raw_model.load(path=rawModel_path, device=device)
    if args.dataset == 'IMAGENET10':
        from RawModels.Resnet_for_imagenet import resnet34_imagenet,loadData
        from art.estimators.classification import PyTorchClassifier
        rawModel_path = 'RawModels/IMAGENET10/model/IMAGENET10_raw.pt'
        train_loader,test_loader = loadData(path='./RawModels/IMAGENET10/dataset/',
                                              batch_size=128)

        raw_model = resnet34_imagenet()
        raw_model.load_state_dict(torch.load(rawModel_path))
        raw_model = PyTorchClassifier(raw_model,loss=None,input_shape=(3,224,224),nb_classes=10,device=device)

    if not os.path.exists(rawModel_path):
        print('please train model first!')


    attack_type = args.attack
    dataset_type = args.dataset
    detect_type= args.type

    advloader = adv_loader(attack_type, dataset_type, 100)
    # feature_path = 'Feature/{}_{}_{}.npy'.format(args.dataset, args.attack, args.type)
    attack_type_list = ['FGSM','PGD','DEEPFOOL','JSMA','CWL0','CWL2','CWLINF']
    test_attack_type_list = ['FGSM','PGD','DEEPFOOL','JSMA','CWL0','CWL2','CWLINF']
    for attack_type in attack_type_list:
        for test_attack_type in test_attack_type_list:
        # test_attack_type = attack_type
            feature_path = 'Feature/{}_{}_{}.npy'.format(args.dataset, attack_type, args.type)
            if not os.path.exists(feature_path):

                if detect_type == 'lid':
                    base_feature_path = 'Feature/base_feature_{}.npy'.format(args.dataset)
                    if os.path.isfile(base_feature_path):
                        base_features, base_labels = np.load(base_feature_path, allow_pickle=True)
                    else:

                        base_features, base_labels = get_baseline(args, raw_model, train_loader, device)
                        np.save(base_feature_path, [base_features, base_labels])

                    adv_data, _ = get_data(advloader, 1000)
                    test_feature_all,test_label = get_baseline(args,raw_model,test_loader,device)
                    # print('test feature shape is:',len(test_feature_all[0]))
                    adv_feature_all,adv_label = get_layers_from_standard_dataload(raw_model,advloader,device)
                    # print('adv_feature shape:',len(adv_feature_all[0]))
                    adv_score = get_score_by_lid(base_features, adv_feature_all)
                    test_score = get_score_by_lid(base_features, test_feature_all)
                elif detect_type == 'kd':


                    None_feature()
                    raw_model.to(device)
                    train_label = predict(raw_model, train_loader, device)
                    train_kde_feature = get_feature()
                    None_feature()
                    test_label = predict(raw_model,test_loader,device)
                    test_kde_feature = get_feature()
                    None_feature()
                    adv_label = predict(raw_model,advloader,device)
                    adv_kde_feature = get_feature()
                    # print('nan:', torch.isnan(adv_kde_feature).any())
                    # print('inf:', torch.isinf(adv_kde_feature).any())
                    test_score, adv_score = get_kde_score(train_kde_feature, train_label, test_kde_feature, test_label,
                                                          adv_kde_feature, adv_label)
                else:
                    print('input right detect type')
                    exit(0)
                np.save(feature_path,[test_score,adv_score])
            else:
                test_score, adv_score = np.load(feature_path,allow_pickle=True)
            # print('test score is :', test_score.mean(0))
            # print('adv score is :',adv_score.mean(0))
            # mix_data, mix_label = mix_feature(test_score, adv_score)
            # test_attack_type=args.test_attack
            if attack_type != test_attack_type:
                # path = './Feature/{}_{}_{}.npy'.format(args.dataset,args.attack,args.type)
                path = './Feature/{}_{}_{}.npy'.format(args.dataset, attack_type, args.type)
                if os.path.exists(path):
                    test_score, adv_score = np.load(path,allow_pickle=True)
                else:
                    print('first generate the feature of attack!')
                    exit()

                mix_data, mix_label = mix_feature(test_score, adv_score)
                print(mix_data.shape)
                scale = MinMaxScaler().fit(mix_data)
                mix_data = scale.transform(mix_data)
                x_train, _, y_train, _ = train_test_split(mix_data, mix_label, random_state=0, test_size=0.33)
                # path1 = './Feature/{}_{}_{}.npy'.format(args.dataset, args.test_attack, args.type)
                path1 = './Feature/{}_{}_{}.npy'.format(args.dataset,test_attack_type,args.type)
                test_score1, adv_score1 = np.load(path1,allow_pickle=True)
                mix_data1, mix_label1 = mix_feature(test_score1, adv_score1)
                _, x_test, _, y_test = train_test_split(mix_data1, mix_label1, random_state=0, test_size=0.33)
            else:
                # path = './Feature/' + detect_type + '_' + attack_type + '.npy'
                # test_score, adv_score = np.load(path)
                mix_data, mix_label = mix_feature(test_score, adv_score)
                x_train, x_test, y_train, y_test = train_test_split(mix_data, mix_label, random_state=0, test_size=0.33)
            x_train_idx = np.isnan(x_train)
            x_test_idx = np.isnan(x_test)
            x_train[x_train_idx] = 0
            x_test[x_test_idx] = 0
            print('nan:', np.isnan(x_train).any())
            print('inf:', np.isinf(x_train).any())
            lr = LogisticRegressionCV(max_iter=1000).fit(x_train, y_train)
            predict_score = lr.predict_proba(x_test)[:, 1]
            plot_auc(y_test, predict_score, attack_type,test_attack_type, dataset_type, detect_type)


if __name__ == '__main__':

    main()
