import torch
import os
import argparse
import numpy as np
import random

from art.estimators.classification import PyTorchClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from torch.nn import Sequential, Linear, BatchNorm1d, Softmax, Dropout

from RawModels.Utils.TrainTest import testing_evaluation, predict
from RawModels.Utils.dataset import get_mnist_test_loader, get_mnist_train_validate_loader,get_svhn_train_validate_loader,get_svhn_test_loader\
    ,get_cifar10_train_validate_loader,get_cifar10_test_loader
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def mle_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    # print('a is :', a.shape)
    return a


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积

    res = num / (denom+1e-10)
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

def get_distance_matrix(v1,v2):
    size=len(v1)
    assert len(v1)==len(v2)
    v1=v1.reshape(size,1,-1)
    v2=v2.reshape(1,size, -1)
    v1_min_v2=v1-v2
    return np.sqrt(np.einsum('ijx,ijx->ij',v1_min_v2,v1_min_v2))


def cos_sim(data, batch, k):
    """
    input: shape is (n,m) n is batch_size,m is dimension
    output:shape is (n,n)
    """
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: np.mean(v)
    a = get_cos_similar_matrix(data, batch)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, -k - 2:-2]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a
def knn_sim(data, base, k):
    """
    input: shape is (n,m) n is batch_size,m is dimension
    output:shape is (n,n)
    """
    data = np.asarray(data, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: np.mean(v)
    a = get_distance_matrix(data, base)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a
def cos_sim_global(data, base, k):
    """
    input: shape is (n,m) n is batch_size,m is dimension
    output:shape is (n,n)
    """
    data = np.asarray(data, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: np.mean(v)
    a = get_cos_similar_matrix(data, base)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a
def cos_sim_in_class(data, label, base, base_label):
    """
    input: shape is (n,m) n is batch_size,m is dimension
    output:shape is (n,n)
    """
    # assert len(data) == len(base)
    n = len(data)
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(base, dtype=np.float32)

    # k = min(k, len(data) - 1)
    a = get_cos_similar_matrix(data, base)
    a = torch.from_numpy(a)
    label=torch.from_numpy(label)
    base_label = torch.from_numpy(base_label)
    # print(a.shape)
    # print('label is :', label.shape)
    # print('adv_label is :', base_label)
    mask = torch.ones_like(a) * (base_label.expand(n, n).eq(label.expand(n, n).t()))
    # duijiao_0 = torch.ones_like(mask) - torch.eye(n)  # 为了代码方便可能丢失对角线上label相同的值，但是对结果不会造成影响。
    # mask = mask * duijiao_0
    # print(torch.sum(mask, 1))
    a = torch.sort(a * mask, 1, descending=True)[0]
    a = torch.sum(a, 1)/torch.sum(mask,1)
    return a.numpy()





def get_lid_by_random(data, adv_data, batch_size, k=10):
    batch_num = int(np.ceil(data.shape[0] / float(batch_size)))
    lid = None
    lid_adv = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(data), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, 1))
        lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = data[start:end]

        X_adv_act = adv_data[start:end]
        lid_batch = mle_batch(X_act, X_act, k)
        lid_batch_adv = mle_batch(X_act, X_adv_act, k)
        if lid is not None:
            lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            lid = lid_batch
            lid_adv = lid_batch_adv
    return lid, lid_adv


def get_ilacs_by_random(data, label1, adv_data, label2, batch_size, k=10):
    batch_num = int(np.ceil(data.shape[0] / float(batch_size)))
    lid = None
    lid_adv = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(data), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, 1))
        lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = data[start:end]
        X_mean = np.mean(X_act, 0)
        X_adv_act = adv_data[start:end]
        lid_batch = cos_sim_in_class(X_act - X_mean, label1, X_act - X_mean, label1)
        lid_batch_adv = cos_sim_in_class(X_act - X_mean, label1, X_adv_act - X_mean, label2)
        if lid is not None:
            lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            lid = lid_batch
            lid_adv = lid_batch_adv
    return lid, lid_adv

def get_ilcs_by_random(data, label1, adv_data, label2, batch_size, k=10):
    batch_num = int(np.ceil(data.shape[0] / float(batch_size)))
    lid = None
    lid_adv = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(data), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, 1))
        lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = data[start:end]
        # X_mean = np.mean(X_act, 0)
        X_adv_act = adv_data[start:end]
        lid_batch = cos_sim_in_class(X_act , label1, X_act , label1)
        lid_batch_adv = cos_sim_in_class(X_act , label1, X_adv_act , label2)
        if lid is not None:
            lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            lid = lid_batch
            lid_adv = lid_batch_adv
    return lid, lid_adv


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




# def get_iacs_feature(data,label,batch_size):
#     batch_num = int(np.ceil(data.shape[0]/float(batch_size)))
#
#     feature = None
#     for i_batch in range(batch_num):
#         start = i_batch*batch_size
#         end = np.min((i_batch+1)*batch_size,len(data))
#         X_data = data[start:end]
#         X_label = label[start:end]
#         X_mean = np.mean(X_data,0)
#         n_feed = end - start
#         feature_batch = np.zeros(shape=(n_feed,1))
#         feature_batch = cos_sim_in_class_global(X_data-X_mean,X_label,X_data-X_mean,X_label)
#
#         if feature is not None:
#             feature = np.concatenate((feature,feature_batch),0)
#         else:
#             feature = feature_batch
#
#     return feature


def get_iacs_feature(test_batch, baseline_batch, test_labels, baseline_labels, batch_size=1000):
    batch_num = int(np.ceil(test_batch.shape[0] / float(batch_size)))

    test_feature = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum((i_batch + 1) * batch_size, len(test_batch))
        n_feed = end - start
        test_data = test_batch[start:end]
        test_label = test_labels[start:end]
        baseline_batch = baseline_batch[0:n_feed]
        baseline_label= baseline_labels[0:n_feed]
        X_mean = np.mean(baseline_batch, 0)

        print(test_label.shape)
        print(baseline_label.shape)
        # feature_batch1 = np.zeros(shape=(n_feed, 1))
        # feature_batch2 = np.zeros(shape=(n_feed, 1))
        feature_batch1 = cos_sim_in_class(test_data - X_mean, test_label, baseline_batch - X_mean,
                                                 baseline_label)
        # feature_batch2 = cos_sim_in_class(test_data - X_mean, test_label, test_data - X_mean, test_label)

        if test_feature is not None:
            test_feature = np.concatenate((test_feature, feature_batch1), 0)
            # test_feature = np.concatenate((test_feature, feature_batch2))
        else:
            test_feature = feature_batch1
            # test_feature = feature_batch2
    print(test_feature.shape)
    return test_feature

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
def process(Y):
    for i in range(len(Y)):
        if Y[i] == 1:
            Y[i] = 0
        elif Y[i] == -1:
            Y[i] = 1
    return Y

def dense(input_shape):
    model = Sequential(
        Dropout(0.2),
        Linear(input_shape, 10),
        # Linear(1000,10),
        # BatchNorm1d(10),
        Softmax(dim=1),
    )
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
def get_train_features(args,features,labels,nb_layer,device,rand=0):
    dir_path = 'detect_train/layers/'
    out_feature ={}
    for l_indx in range(0, nb_layer):
        full_connect_model_path = '{}{}_{}.pth'.format(dir_path, args.dataset, l_indx)
        # layer_data_path = '{}{}_{}_normal.npy'.format(dir_path, args.dataset, l_indx)
        # if not os.path.isfile(layer_data_path):
        #     os.makedirs(dir_path,exist_ok=True)
        #     l_out = features[l_indx]
        #     np.save(layer_data_path, l_out)
        # else:
        #     l_out= np.load(layer_data_path, allow_pickle=True)
        l_out = features[l_indx]
        current_model = dense(input_shape=len(l_out[0]))
        current_model.load_state_dict(torch.load(full_connect_model_path))
        current_model.eval()
        num = len(l_out)
        epoch = 100

        iter = int(np.ceil(num/float(epoch)))
        for i in range(iter):
            start = i*epoch
            end = np.minimum(num,(i+1)*epoch)

            tem = l_out[start:end]
            if isinstance(tem,np.ndarray):
                tem = torch.from_numpy(tem)
            if out_feature.get(l_indx,None) is None:
                out_feature[l_indx] = current_model(tem).detach().numpy()
            else:
                # print('success')
                out_feature[l_indx] = np.concatenate((out_feature[l_indx],current_model(tem).detach().numpy()))

    return out_feature,labels
def get_train_model(args,train_features,labels,nb_layer,device):
    train_dir = 'detect_train/layers/'
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir,exist_ok=True)
    for l_indx in range(0,nb_layer):
        full_connect_model_path = '{}{}_{}.pth'.format(train_dir, args.dataset, l_indx)
        layer_data_path = '{}{}_{}_normal.npy'.format(train_dir,args.dataset,l_indx)
        # if not os.path.isfile(layer_data_path):
        l_out = train_features[l_indx]
        # np.save(layer_data_path,l_out)
        # else:
        #     l_out = np.load(layer_data_path,allow_pickle=True)
        current_model = dense(input_shape=len(l_out[0]))
        print(l_out.shape)
        print(labels.shape)
        train(current_model,l_out,labels,args)
        torch.save(current_model.state_dict(),full_connect_model_path)
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
def get_score_by_KNN(dict1, dict2):
    def get_knn_by_random(base_data, adv_data, batch_size, k=10):
        def knn_sim(data, batch, k):

            """
            input: shape is (n,m) n is batch_size,m is dimension
            output:shape is (n,n)
            """
            data = np.asarray(data, dtype=np.float32)
            batch = np.asarray(batch, dtype=np.float32)

            k = min(k, len(data) - 1)
            f = lambda v: np.mean(v)
            a = get_distance_matrix(data, batch)
            a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
            a = np.apply_along_axis(f, axis=1, arr=a)
            return a
        batch_num = int(np.ceil(adv_data.shape[0] / float(batch_size)))
        lid = None
        lid_adv = None
        for i_batch in range(batch_num):
            start = i_batch * batch_size
            end = np.minimum(len(adv_data), (i_batch + 1) * batch_size)
            n_feed = end - start
            # lid_batch = np.zeros(shape=(n_feed, 1))
            # lid_batch_adv = np.zeros(shape=(n_feed, 1))
            X_act = base_data[0:n_feed]
            # X_mean = np.mean(X_act, 0)
            X_adv_act = adv_data[start:end]
            # lid_batch = knn_sim(X_act,  X_act ,k)
            lid_batch_adv = knn_sim(X_adv_act, X_act, k)
            if lid_adv is not None:
                # lid = np.concatenate((lid, lid_batch), 0)
                lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
            else:
                # lid = lid_batch
                lid_adv = lid_batch_adv
        return lid_adv
    # assert len(dict1) == len(dict2)
    # print(dict1[0][0])
    # print('####################################')
    # print(dict2[0][0])
    # print('####################################')
    dim = len(dict1)
    # base_score = np.zeros((len(dict1[0]), dim))
    adv_score = np.zeros((len(dict2[0]), dim))
    for i in range(dim):
        if isinstance(dict1[i], np.ndarray):
            feature1 = dict1[i]
        else:
            feature1 = dict1[i].cpu().detach().numpy()
            # print('feature1 is :', feature1[0])
        if isinstance(dict2[i], np.ndarray):
            feature2 = dict2[i]
        else:
            feature2 = dict2[i].cpu().detach().numpy()
        # print('feature2 is :', feature2[0])


        adv_score[:, i] = get_knn_by_random(feature1, feature2, batch_size=100, k=10)

    return adv_score

def get_score_by_ED(dict1, dict2):
    def get_knn_by_random(base_data, adv_data, batch_size, k=10):
        def ed_sim(data, batch, k):

            """
            input: shape is (n,m) n is batch_size,m is dimension
            output:shape is (n,n)
            """
            data = np.asarray(data, dtype=np.float32)
            batch = np.asarray(batch, dtype=np.float32)

            k = min(k, len(data) - 1)
            f = lambda v: np.mean(v)
            a = get_distance_matrix(data, batch)
            a = np.apply_along_axis(np.sort, axis=1, arr=a)
            a = np.apply_along_axis(f, axis=1, arr=a)
            return a
        batch_num = int(np.ceil(adv_data.shape[0] / float(batch_size)))
        lid = None
        lid_adv = None
        for i_batch in range(batch_num):
            start = i_batch * batch_size
            end = np.minimum(len(adv_data), (i_batch + 1) * batch_size)
            n_feed = end - start
            # lid_batch = np.zeros(shape=(n_feed, 1))
            # lid_batch_adv = np.zeros(shape=(n_feed, 1))
            X_act = base_data[0:n_feed]
            # X_mean = np.mean(X_act, 0)
            X_adv_act = adv_data[start:end]
            # lid_batch = knn_sim(X_act,  X_act ,k)
            lid_batch_adv = ed_sim(X_adv_act, X_act, k)
            if lid_adv is not None:
                # lid = np.concatenate((lid, lid_batch), 0)
                lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
            else:
                # lid = lid_batch
                lid_adv = lid_batch_adv
        return lid_adv
    # assert len(dict1) == len(dict2)
    # print(dict1[0][0])
    # print('####################################')
    # print(dict2[0][0])
    # print('####################################')
    dim = len(dict1)
    # base_score = np.zeros((len(dict1[0]), dim))
    adv_score = np.zeros((len(dict2[0]), dim))
    for i in range(dim):
        if isinstance(dict1[i], np.ndarray):
            feature1 = dict1[i]
        else:
            feature1 = dict1[i].cpu().detach().numpy()
            # print('feature1 is :', feature1[0])
        if isinstance(dict2[i], np.ndarray):
            feature2 = dict2[i]
        else:
            feature2 = dict2[i].cpu().detach().numpy()
        # print('feature2 is :', feature2[0])


        adv_score[:, i] = get_knn_by_random(feature1, feature2, batch_size=100, k=10)

    return adv_score
def get_score_by_IED(base_features, base_labels, test_features, test_labels,batch_size=1000):
    # assert len(dict1) == len(dict2)
    # print(dict1[0][0])
    # print('####################################')
    # print(dict2[0][0])
    # print('####################################')
    dim = len(test_features)
    # base_score = np.zeros((len(dict1[0]), dim))
    adv_score = np.zeros((len(test_features[0]), dim))
    # print(adv_score.shape)
    for i in range(dim):

        if isinstance(base_features[i], np.ndarray):
            feature1 = base_features[i]
        if isinstance(base_features[i], torch.Tensor):
            feature1 = base_features[i].cpu().detach().numpy()

        # print('feature1 is :', feature1[0])
        if isinstance(test_features[i], np.ndarray):
            feature2 = test_features[i]
        if isinstance(test_features[i], torch.Tensor):
            feature2 = test_features[i].cpu().detach().numpy()
        # print('feature2 is :', feature2[0])

        adv_score[:, i] = get_ied_by_random(feature1, base_labels, feature2, test_labels, batch_size=1000, k=10)
    return adv_score
def ied_in_class_global(param, label1, param1, label2):
    """
           input: shape is (n,m) n is batch_size,m is dimension
           output:shape is (n,n)
           """
    assert len(param) == len(param1)
    n = len(param)

    data = np.asarray(param, dtype=np.float32)
    batch = np.asarray(param1, dtype=np.float32)

    a = get_distance_matrix(batch, data)
    a = torch.from_numpy(a)

    if isinstance(label1, np.ndarray):
        label = torch.from_numpy(label1)
    if isinstance(label2, np.ndarray):
        batch_label = torch.from_numpy(label2)
    mask = torch.ones_like(a) * (label.expand(n, n).eq(batch_label.expand(n, n).t()))
    # duijiao_0 = torch.ones_like(mask) - torch.eye(n)  # 为了代码方便可能丢失对角线上label相同的值，但是对结果不会造成影响。
    # mask = mask * duijiao_0
    # print(torch.sum(mask, 1))
    a = torch.sort(a * mask, 1, descending=True)[0]
    a = torch.sum(a, 1) / torch.sum(mask, 1)
    return a.numpy()

def get_ied_by_random(feature1, base_labels, feature2, test_labels, batch_size, k):
    batch_num = int(np.ceil(feature2.shape[0] / float(batch_size)))
    lid_adv = None
    # print(len(label2))
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(feature2), (i_batch + 1) * batch_size)
        n_feed = end - start
        # lid_batch = np.zeros(shape=(n_feed, 1))
        # lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = feature1[0:n_feed]
        label1 = base_labels[0:n_feed]
        X_mean = np.mean(X_act, 0)
        X_adv_act = feature2[start:end]
        label2 = test_labels[start:end]
        X_adv_mean = np.mean(X_adv_act, 0)
        # print(len(label1))
        # print(len(label2))
        print('feature1:', len(X_act))
        print('feature2:', len(X_adv_act))
        lid_batch_adv = ied_in_class_global(X_act - X_mean, label1, X_adv_act - X_adv_mean, label2)
        if lid_adv is not None:
            # lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            # lid = lid_batch
            lid_adv = lid_batch_adv
    return lid_adv
def detect(args,model,train_load,test_load,adv_load,nb_layer,device,rand):
    """
    step1: get iacs features of train set
    step2: training a oneclass_SVM
    step3: get iacs features of adversarial set
    step4: output the result
    """
    import sklearn.svm as svm
    import time
    import pickle
    import random
    svm_model_path = 'unsupervise/{}_{}_svm.model'.format(args.dataset,args.type)
    layers_data_baseline_path = 'unsupervise/{}_layers_baseline.npy'.format(args.dataset)
    layers_data_train_path = 'unsupervise/{}_layers_trian.npy'.format(args.dataset)
    layers_data_test_path = 'unsupervise/{}_layers_test.npy'.format(args.dataset)
    layers_data_adv_path = 'unsupervise/{}_layers_{}.npy'.format(args.dataset,args.attack)
    if not os.path.isfile(layers_data_test_path):
        test_layers_data, test_labels = get_layers_feature(model, test_load, nb_layer, device,rand)
        np.save(layers_data_test_path,[test_layers_data,test_labels])
    else:
        test_layers_data, test_labels = np.load(layers_data_test_path,allow_pickle=True)
    if not os.path.isfile(layers_data_adv_path):
        adv_layers_data, adv_labels = get_layers_feature(model, adv_load,nb_layer, device,rand=0)
        np.save(layers_data_adv_path, [adv_layers_data, adv_labels])
    else:
        adv_layers_data, adv_labels = np.load(layers_data_adv_path,allow_pickle=True)
    if not os.path.isfile(layers_data_train_path):
        layers_data_train, train_labels = get_layers_feature(model, train_load, nb_layer, device=device, rand=0)
        np.save(layers_data_train_path, [layers_data_train, train_labels])
    else:
        layers_data_train, train_labels = np.load(layers_data_train_path, allow_pickle=True)
    # get iacs feature
    lengh = len(layers_data_train[0])
    dim = len(layers_data_train)
    if not os.path.isfile(layers_data_baseline_path):

        layers_data_baseline, labels_baseline = get_layers_feature(model, train_load, nb_layer, device, rand=rand)
        np.save(layers_data_baseline_path, [layers_data_baseline, labels_baseline])
    else:
        layers_data_baseline, labels_baseline = np.load(layers_data_baseline_path, allow_pickle=True)
    # train_features = np.zeros((lengh,dim))
    if args.train:

        get_train_model(args, layers_data_train, train_labels, nb_layer, device)
    test_data, test_labels = get_train_features(args, test_layers_data, test_labels, nb_layer, device)
    adv_data, adv_labels = get_train_features(args, adv_layers_data, adv_labels, nb_layer, device)
    print('adv shape :', adv_data[0].shape)
    base_data, base_labels = get_train_features(args, layers_data_baseline, labels_baseline, nb_layer, device)
    train_data, train_labels = get_train_features(args, layers_data_train, train_labels, nb_layer, device)
    if not os.path.isfile(svm_model_path):



        if args.type == 'ied':
            train_features = get_score_by_IED(base_data,base_labels,train_data,train_labels)
        if args.type == 'knn':
            train_features = get_score_by_KNN(base_data,  train_data)
        if args.type == 'ics':
            train_features = get_score_by_ICS(base_data, base_labels, train_data, train_labels)
        if args.type == 'iacs':
            train_features = get_score_by_IACS(base_data, base_labels, train_data, train_labels)
        # train_features = get_score_by_ED(base_data, train_data)
        # print(base_labels.shape)
        # print(train_labels.shape)
        # train_features = get_score_by_IACS(base_data,base_labels,train_data,train_labels)
        clf = svm.OneClassSVM(nu=0.1,kernel='rbf',gamma='scale',verbose=True)
        st = time.time()
        clf.fit(train_features)
        s = pickle.dumps(clf)
        f = open(svm_model_path,"wb+")
        f.write(s)
        f.close()
        et = time.time()
        print("Training svm modle on {} is complete on {} min(s)".format(args.dataset,et-st))
    else:
        clf = pickle.load(open(svm_model_path,'rb'))
    # get iacs features for test and adv data
    # test_layers_data, test_labels = get_layers_from_standard_dataload(model, test_load, device)
    # np.save(layers_data_test_path, [test_layers_data, test_labels])
    # adv_layers_data, adv_labels = get_layers_from_standard_dataload(model, adv_load, device)
    # np.save(layers_data_adv_path, [adv_layers_d
    # ata, adv_labels])

    # adv_lengh = len(adv_layers_data[0])
    # dim = len(adv_layers_data)
    # adv_features = np.zeros((adv_lengh, dim))
    # test_features = np.zeros((adv_lengh, dim))
    # if os.path.isfile(layers_data_baseline_path):
    #     layers_data_baseline,labels_baseline = np.load(layers_data_baseline_path,allow_pickle=True)
    # else:
    #     print('layers_data_baseline is not existing!')
    if args.type == 'ied':
        test_features = get_score_by_IED(base_data,base_labels,test_data,test_labels)
        adv_features =get_score_by_IED(base_data,base_labels,adv_data,adv_labels)
    if args.type == 'knn':
        test_features = get_score_by_KNN(base_data,  test_data)
        adv_features = get_score_by_KNN(base_data,  adv_data)
    if args.type == 'ics':
        # test_features = get_score_by_ED(base_data,  test_data)
        # adv_features = get_score_by_ED(base_data,  adv_data)
        test_features = get_score_by_ICS(base_data, base_labels, test_data, test_labels)
        adv_features = get_score_by_ICS(base_data, base_labels, adv_data, adv_labels)
    if args.type =='iacs':
        test_features = get_score_by_IACS(base_data, base_labels, test_data, test_labels)
        adv_features = get_score_by_IACS(base_data, base_labels, adv_data, adv_labels)
    print(test_features.mean(0))
    print(adv_features.mean(0))
    adv_pred = clf.predict(adv_features)
    test_pred = clf.predict(test_features)
    Y_predict = np.concatenate((adv_pred, test_pred))
    adv_prob = clf.decision_function(adv_features)
    test_prob = clf.decision_function(test_features)
    Y_prob = np.concatenate((adv_prob, test_prob))
    adv_labels = np.ones((len(adv_pred)))
    test_labels = np.zeros((len(test_pred)))
    Y_label = np.concatenate((adv_labels, test_labels))
    Y_predict = process(Y_predict)
    acc, tpr, fpr, tp, ap, fp, an = evalulate_detection_test(Y_label, Y_predict)
    fprs_all, tprs_all, thresholds_all = roc_curve(Y_label, -Y_prob)
    roc_auc = auc(fprs_all, tprs_all)

    print("attck {} ACC: {:.4f}%, TPR :{:.4f}% FPR : {:.4f}% AUC : {:.4f}%".format(args.attack,acc*100,tpr*100,fpr*100,roc_auc*100))
def get_score_by_IACS(base_features, base_labels, adv_features, adv_labels):
    # assert len(dict1) == len(dict2)
    # print(dict1[0][0])
    # print('####################################')
    # print(dict2[0][0])
    # print('####################################')
    dim = len(adv_features)
    # base_score = np.zeros((len(dict1[0]), dim))
    adv_score = np.zeros((len(adv_features[0]), dim))
    # print(adv_score.shape)
    for i in range(dim):

        if isinstance(base_features[i], np.ndarray):
            feature1 = base_features[i]
        if isinstance(base_features[i], torch.Tensor):
            feature1 = base_features[i].cpu().detach().numpy()

        # print('feature1 is :', feature1[0])
        if isinstance(adv_features[i], np.ndarray):
            feature2 = adv_features[i]
        if isinstance(adv_features[i], torch.Tensor):
            feature2 = adv_features[i].cpu().detach().numpy()
        # print('feature2 is :', feature2[0])

        adv_score[:, i] = get_iacs_by_random(feature1, base_labels, feature2, adv_labels, batch_size=1000, k=10)
    return adv_score

def get_score_by_ICS(base_features, base_labels, adv_features, adv_labels):
    # assert len(dict1) == len(dict2)
    # print(dict1[0][0])
    # print('####################################')
    # print(dict2[0][0])
    # print('####################################')
    dim = len(adv_features)
    # base_score = np.zeros((len(dict1[0]), dim))
    adv_score = np.zeros((len(adv_features[0]), dim))
    # print(adv_score.shape)
    for i in range(dim):

        if isinstance(base_features[i], np.ndarray):
            feature1 = base_features[i]
        if isinstance(base_features[i], torch.Tensor):
            feature1 = base_features[i].cpu().detach().numpy()

        # print('feature1 is :', feature1[0])
        if isinstance(adv_features[i], np.ndarray):
            feature2 = adv_features[i]
        if isinstance(adv_features[i], torch.Tensor):
            feature2 = adv_features[i].cpu().detach().numpy()
        # print('feature2 is :', feature2[0])

        adv_score[:, i] = get_ics_by_random(feature1, base_labels, feature2, adv_labels, batch_size=1000, k=10)
    return adv_score

def get_ics_by_random(feature1, label1, feature2, label2, batch_size, k):
    batch_num = int(np.ceil(feature2.shape[0] / float(batch_size)))
    # lid = None
    lid_adv = None
    # print(len(label2))
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(feature2), (i_batch + 1) * batch_size)
        n_feed = end - start
        # lid_batch = np.zeros(shape=(n_feed, 1))
        # lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = feature1[0:n_feed]
        label1 = label1[0:n_feed]
        X_adv_act = feature2[start:end]
        label_adv = label2[start:end]
        X_adv_mean = np.mean(X_adv_act,0)
        # print(len(label1))
        # print(len(label2))
        # print('feature1:',len(X_act))
        # print('feature2:',len(X_adv_act))
        print(label1.shape)
        print(label2.shape)
        lid_batch_adv = cos_sim_in_class_global(X_act , label1, X_adv_act , label_adv)
        if lid_adv is not None:
            # lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            # lid = lid_batch
            lid_adv = lid_batch_adv
    return lid_adv

def get_iacs_by_random(feature1, label1, feature2, label2, batch_size, k):
    batch_num = int(np.ceil(feature2.shape[0] / float(batch_size)))
    # lid = None
    lid_adv = None
    # print(len(label2))
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(feature2), (i_batch + 1) * batch_size)
        n_feed = end - start
        # lid_batch = np.zeros(shape=(n_feed, 1))
        # lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = feature1[0:n_feed]
        label1 = label1[0:n_feed]
        X_mean = np.mean(X_act, 0)
        X_adv_act = feature2[start:end]
        label_adv = label2[start:end]
        X_adv_mean = np.mean(X_adv_act,0)
        # print(len(label1))
        # print(len(label2))
        # print('feature1:',len(X_act))
        # print('feature2:',len(X_adv_act))
        print(label1.shape)
        print(label2.shape)
        lid_batch_adv = cos_sim_in_class_global(X_act - X_mean, label1, X_adv_act - X_adv_mean, label_adv)
        if lid_adv is not None:
            # lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            # lid = lid_batch
            lid_adv = lid_batch_adv
    return lid_adv

def cos_sim_in_class_global(param, label1, param1, label2):
    """
       input: shape is (n,m) n is batch_size,m is dimension
       output:shape is (n,n)
       """
    assert len(param) == len(param1)
    n = len(param)

    data = np.asarray(param, dtype=np.float32)
    batch = np.asarray(param1, dtype=np.float32)


    a = get_cos_similar_matrix(batch, data)
    a = torch.from_numpy(a)

    if isinstance(label1, np.ndarray):
        label = torch.from_numpy(label1)
    if isinstance(label2, np.ndarray):
        batch_label = torch.from_numpy(label2)
    mask = torch.ones_like(a) * (label.expand(n, n).eq(batch_label.expand(n, n).t()))
    # duijiao_0 = torch.ones_like(mask) - torch.eye(n)  # 为了代码方便可能丢失对角线上label相同的值，但是对结果不会造成影响。
    # mask = mask * duijiao_0
    # print(torch.sum(mask, 1))
    a = torch.sort(a * mask, 1, descending=True)[0]
    a = torch.sum(a, 1) / torch.sum(mask, 1)
    return a.numpy()

def main():
    import argparse
    parse= argparse.ArgumentParser(description='this is an adversarial detection based on Local Cosine Similarity')
    parse.add_argument('--seed', type=int, default=100, help='set the random seed')
    parse.add_argument('--gpu_index', type=str, default='1', help='the index of gpu')
    parse.add_argument('--dataset',default='MNIST',choices=['MNIST','SVHN','CIFAR10'])
    parse.add_argument('--type',default='cs',choices=['knn','ied','iacs','ics'])
    parse.add_argument('--attack',default='FGSM',choices=['FGSM','PGD','DEEPFOOL','JSMA','CWL0','CWL2','CWLINF'])
    parse.add_argument('--test_attack', default='fgsm', choices=['FGSM','PGD','DEEPFOOL','JSMA','CWL0','CWL2','CWLINF'])
    parse.add_argument('--batch_size',default=128)
    parse.add_argument('--train', default = False)
    parse.add_argument('--epochs',default= 30)
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

    if args.dataset == 'MNIST':
        from RawModels.MNISTConv import MNISTConvNet, MNIST_Training_Parameters, None_feature, get_feature
        from RawModels.Utils.dataset import get_mnist_test_loader, get_mnist_train_validate_loader
        rawModel_path = 'RawModels/MNIST/model/MNIST_raw.pt'

        test_loader = get_mnist_test_loader(dir_name='./RawModels/MNIST/',
                                            batch_size=MNIST_Training_Parameters['batch_size'])
        train_loader, _, _ = get_mnist_train_validate_loader(dir_name='./RawModels/MNIST/',
                                                             batch_size=MNIST_Training_Parameters['batch_size'])
        raw_model = MNISTConvNet().to(device)
        raw_model.load(path=rawModel_path, device=device)
        raw_model = PyTorchClassifier(raw_model, loss=None, input_shape=(1, 28, 28), nb_classes=10)
        nb_layer = 6
        rand = 1000

    if args.dataset == 'SVHN':
        from RawModels.SVHNConv import SVHNConvNet, SVHN_Training_Parameters, None_feature, get_feature
        from RawModels.Utils.dataset import get_svhn_test_loader, get_svhn_train_validate_loader
        rawModel_path = 'RawModels/SVHN/model/SVHN_raw.pt'
        test_loader = get_svhn_test_loader(dir_name='./RawModels/SVHN/',
                                           batch_size=SVHN_Training_Parameters['batch_size'])
        train_loader, _, _ = get_svhn_train_validate_loader(dir_name='./RawModels/SVHN/',
                                                            batch_size=SVHN_Training_Parameters['batch_size'])
        raw_model = SVHNConvNet().to(device)
        raw_model.load(path=rawModel_path, device=device)
        raw_model = PyTorchClassifier(raw_model, loss=None, input_shape=(3, 32, 32), nb_classes=10)
        nb_layer = 6
        rand = 1000
    if args.dataset == 'CIFAR10':
        from RawModels.ResNet import resnet20_cifar, CIFAR10_Training_Parameters, None_feature, get_feature
        from RawModels.Utils.dataset import get_cifar10_test_loader, get_cifar10_train_validate_loader
        rawModel_path = 'RawModels/CIFAR10/model/CIFAR10_raw.pt'
        test_loader = get_cifar10_test_loader(dir_name='./RawModels/CIFAR10/',
                                              batch_size=CIFAR10_Training_Parameters['batch_size'])
        train_loader, _, _ = get_cifar10_train_validate_loader(dir_name='./RawModels/CIFAR10/',
                                                               batch_size=CIFAR10_Training_Parameters['batch_size'])
        raw_model = resnet20_cifar().to(device)
        raw_model.load(path=rawModel_path, device=device)
        raw_model = PyTorchClassifier(raw_model, loss=None, input_shape=(3, 32, 32), nb_classes=10)
        nb_layer = 9
        rand = 1000
    if args.dataset == 'IMAGENET10':
        from RawModels.Resnet_for_imagenet import resnet34_imagenet, loadData

        rawModel_path = 'RawModels/IMAGENET10/model/IMAGENET10_raw.pt'
        train_loader, test_loader = loadData(path='./RawModels/IMAGENET10/dataset/',
                                             batch_size=32)

        raw_model = resnet34_imagenet()
        raw_model.load_state_dict(torch.load(rawModel_path))
        raw_model.to(device)
        raw_model = PyTorchClassifier(raw_model, loss=None, input_shape=(3, 224, 224), nb_classes=10)
        nb_layer = 9
        rand = 400
    if not os.path.exists(rawModel_path):
        print('please train model first!')

    # raw_model.load(path=rawModel_path, device=device)
    attack_list = ['FGSM', 'PGD', 'DEEPFOOL', 'JSMA', 'CWL0', 'CWL2', 'CWLINF']
    # test_attack_list = ['FGSM','PGD','DEEPFOOL','JSMA','CWL0','CWL2','CWLINF']
    for attack in attack_list:

        args.attack = attack
        adv_load = adv_loader(args.attack,args.dataset,batch_size=args.batch_size)
        detect(args,raw_model,train_load=train_loader,test_load=test_loader,adv_load=adv_load,nb_layer=nb_layer,device=device,rand=rand)


if __name__ == '__main__':

    main()
