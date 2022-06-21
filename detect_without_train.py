import os
import random

import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from art.estimators.classification import PyTorchClassifier
from torch.nn import Sequential, Linear, Softmax,BatchNorm1d


def dense(input_shape):
    model = Sequential(Linear(input_shape,10),
                       BatchNorm1d(10),
                       Softmax(dim=1))
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def adv_loader(attack, dataset, batch_size):
    from torch.utils.data import TensorDataset, DataLoader
    advdata_path = './AdversarialExampleDatasets/{}/{}/{}_AdvExamples.npy'.format(attack, dataset, attack)
    Advlabel_path = './AdversarialExampleDatasets/{}/{}/{}_AdvLabels.npy'.format(attack, dataset, attack)
    Turelabel_path = './AdversarialExampleDatasets/{}/{}/{}_TrueLabels.npy'.format(attack, dataset, attack)
    adv_data = np.load(advdata_path)
    adv_label = np.load(Advlabel_path)
    true_label = np.argmax(np.load(Turelabel_path),1)

    cor_ind = np.not_equal(adv_label,true_label)
    # print(true_label)
    print(adv_label)
    adv_data = torch.from_numpy(adv_data[cor_ind])
    adv_label = torch.from_numpy(adv_label[cor_ind])
    adv_data = adv_data.to(torch.float32)
    # _, true_label = torch.max(true_label, 1)
    # print(true_label.shape)
    adv_dataset = TensorDataset(adv_data, adv_label)
    adv_loader = DataLoader(adv_dataset, batch_size=batch_size)
    print('The successful attack number is :{}'.format(len(adv_label)))
    return adv_loader


# def get_layers_feature(model,data_loader,nb_layer,device,rand):
#
#     import random
#     layers_feature = {}
#     feature = {}
#     label = np.array([])
#     projector = PCA(n_components=3000)
#     index = False
#     for iter,data in enumerate(data_loader):
#         x, y = data
#         x, y = x.to(device), y.to(device)
#         output = model.model(x)
#         cor_ind = torch.argmax(output, 1).eq(y).cpu().numpy()
#         x = x.cpu().numpy()
#         # print(type(x))
#         for i in range(nb_layer):
#             if feature.get(i,None) is None:
#                 tem= model.get_activations(x, i).reshape(len(x),-1)
#                 # print(type(tem))
#                 tem = np.asarray(tem,dtype='float32')
#                 feature[i]=tem[cor_ind]
#
#
#             else:
#                 tem = np.asarray(model.get_activations(x, i).reshape(len(x), -1),dtype='float32')
#                 print(feature[i].shape)
#                 print(tem.shape)
#                 # tem = model.get_activations(x, i).reshape(len(x), -1)
#                 feature[i] = np.concatenate((feature[i],tem[cor_ind]
#                                                     ))
#             if iter % 130 == 0 and iter != 0:
#                 if feature[i].shape[1]>3000:
#                     print(feature[i].shape[0])
#                     index = True
#                     reduce_feature = projector.fit_transform(feature[i])
#                     if layers_feature.get(i,None) is None:
#                         layers_feature[i]=reduce_feature
#                     else:
#                         layers_feature[i] = np.concatenate((layers_feature[i],reduce_feature))
#                 else:
#                     if layers_feature.get(i,None) is None:
#                         layers_feature[i]=feature[i]
#                     else:
#                         layers_feature[i] = np.concatenate((layers_feature[i],feature[i]))
#                 feature[i] = None
#         if label.shape == 0:
#             label = y[cor_ind].cpu().numpy()
#         else:
#             label = np.concatenate((label, y[cor_ind].cpu().numpy()))
#     if index:
#         num = layers_feature[0].shape[0]
#         label = label[0:num]
#     if rand != 0:
#         ind = random.sample(range(len(label)),rand)
#         for i in range(nb_layer):
#             layers_feature[i]= layers_feature[i][ind]
#         label = label[ind]
#
#     return layers_feature,label

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

def get_lid_by_random(data, adv_data, batch_size, k=10):

    batch_num = int(np.ceil(adv_data.shape[0] / float(batch_size)))
    lid = None
    # lid_adv = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(adv_data), (i_batch + 1) * batch_size)
        n_feed = end - start
        # lid_batch = np.zeros(shape=(n_feed, 1))
        # lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = data[0:n_feed]

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
def mle_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)
    print(data.shape)
    print(batch.shape)
    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    # print('a is :', a.shape)
    return a
def cos_sim_in_class(data, label, batch, batch_label):
    """
    input: shape is (n,m) n is batch_size,m is dimension
    output:shape is (n,n)
    """
    assert len(data) == len(batch)
    n = len(data)
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    # k = min(k, len(data) - 1)
    a = get_cos_similar_matrix(batch, data)
    a = torch.from_numpy(a)
    label= torch.from_numpy(label)
    # print(a.shape)
    # print(label.shape)
    # print(batch_label.shape)
    # batch_label = torch.from_numpy(batch_label)
    # print(a.shape)
    # print('label is :', label)
    # print('adv_label is :', batch_label)
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    if isinstance(batch_label, np.ndarray):
        batch_label = torch.from_numpy(batch_label)
    mask = torch.ones_like(a) * (batch_label.expand(n, n).eq(label.expand(n, n).t()))
    # duijiao_0 = torch.ones_like(mask) - torch.eye(n)  # 为了代码方便可能丢失对角线上label相同的值，但是对结果不会造成影响。
    # mask = mask * duijiao_0
    # print(torch.sum(mask, 1))
    a = torch.sort(a * mask, 1, descending=True)[0][:, 0:10]
    a = torch.mean(a, 1)
    return a.numpy()

def get_score_by_ILACS(dict1, label1, dict2, label2):
    def get_ilacs_by_random(data, label1, adv_data, label2, batch_size, k=10):

        batch_num = int(np.ceil(data.shape[0] / float(batch_size)))
        feature = None
        # lid_adv = None
        for i_batch in range(batch_num):
            start = i_batch * batch_size
            end = np.minimum(len(adv_data), (i_batch + 1) * batch_size)
            n_feed = end - start
            X_act = data[0:n_feed]
            label1 = label1[0:n_feed]
            X_mean = np.mean(data, 0)
            X_adv_act = adv_data[start:end]
            label2 = label2[start:end]
            lid_batch = cos_sim_in_class(X_act - X_mean, label1, X_adv_act - X_mean, label2)
            # lid_batch_adv = cos_sim_in_class(X_act - X_mean, label1, X_adv_act - X_mean, label2)
            if feature is not None:
                feature = np.concatenate((feature, lid_batch), 0)
                # lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
            else:
                feature = lid_batch
                # lid_adv = lid_batch_adv
        return feature
    # assert len(dict1) == len(dict2)
    # print(dict1[0][0])
    # print('####################################')
    # print(dict2[0][0])
    # print('####################################')
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

        adv_score[:, i] = get_ilacs_by_random(feature1, label1, feature2, label2, batch_size=1000, k=10)
    return adv_score


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积

    res = num / (denom+1e-10)
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


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


def get_iacs_by_random(feature1, label1, feature2, label2, batch_size, k):
    batch_num = int(np.ceil(feature2.shape[0] / float(batch_size)))
    lid = None
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
        label2 = label2[start:end]
        X_adv_mean = np.mean(X_adv_act,0)
        # print(len(label1))
        # print(len(label2))
        print('feature1:',len(X_act))
        print('feature2:',len(X_adv_act))
        lid_batch_adv = cos_sim_in_class_global(X_act - X_mean, label1, X_adv_act - X_adv_mean, label2)
        if lid is not None:
            # lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            # lid = lid_batch
            lid_adv = lid_batch_adv
    return lid_adv


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

### kde 方法
def score_point(dup):
    x, kde = dup
    return kde.score_samples(np.reshape(x, (1, -1)))[0]
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
def get_kde_score(train_feature, train_labels, test_feature, test_labels, adv_feature, adv_labels):
    import pickle
    if not isinstance(adv_labels,np.ndarray):
        adv_labels = adv_labels.numpy()
    if not isinstance(test_labels,np.ndarray):
        test_labels = test_labels.numpy()
    class_idx = {}
    for i in range(10):
        class_idx[i] = np.where(train_labels == i)
        print(class_idx[i][0].shape)
    kdes = {}

    for i in range(10):
        kdes[i] = KernelDensity(bandwidth=0.26, kernel='gaussian').fit(train_feature[class_idx[i]],
                                                                       train_labels[class_idx[i]])

    test_score = score_samples(kdes, test_feature, test_labels, n_jobs=1)
    test_score = test_score.reshape((-1, 1))
    adv_ind = np.isnan(adv_feature)
    adv_feature[adv_ind] = 0
    adv_score = score_samples(kdes, adv_feature, adv_labels, n_jobs=1)
    adv_score = adv_score.reshape((-1, 1))

    return test_score, adv_score

def mix_feature(test_score, adv_score):
    adv_label = np.zeros(len(adv_score))
    test_label = np.ones(len(test_score))
    mix_data = np.concatenate((test_score, adv_score), 0)
    mix_label = np.concatenate((test_label, adv_label), 0)
    return mix_data, mix_label


def plot_auc(y_true, y_pre,attack,test_attack,dataset_type,detect_type):
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
    print(' attack {} test attack {} auc is : {}'.format(attack,test_attack,roc_auc))


def get_layer_feature(model, data_loader, nb_layer, device, rand):
    import random
    layers_feature = np.array([])
    label = np.array([])
    for data in data_loader:
        x, y = data
        x, y = x.to(device), y.to(device)
        output = model.model(x)
        cor_ind = torch.argmax(output, 1).eq(y).cpu().numpy()
        x = x.cpu().numpy()
        # print(type(x))

        if layers_feature.shape[0] == 0:
            tem = model.get_activations(x, nb_layer-1).reshape(len(x), -1)
            # print(type(tem))
            layers_feature = tem[cor_ind]


        else:
            tem = model.get_activations(x, nb_layer-1).reshape(len(x), -1)
            layers_feature = np.concatenate((layers_feature, tem[cor_ind]
                                                ))
        if label.shape[0] == 0:
            label = y[cor_ind].cpu().numpy()
        else:
            label = np.concatenate((label, y[cor_ind].cpu().numpy()))
    if rand != 0:
        ind = random.sample(range(len(label)), rand)

        layers_feature = layers_feature[ind]
        label = label[ind]

    return layers_feature, label

def tsne(X,Y):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    Tsne =TSNE(n_components=2)
    X_tsne = Tsne.fit_transform(X)
    plt.scatter(X_tsne[:,0],X_tsne[:,1],c=Y)
    plt.show()
    return X_tsne,Y

def tsne_(X):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    Tsne =TSNE(n_components=1)
    X_tsne = Tsne.fit_transform(X)
    # plt.scatter(X_tsne[:,0],X_tsne[:,1],c=Y)
    # plt.show()
    return X_tsne


def get_distance_matrix(v1, v2):
    size = len(v1)
    assert len(v1) == len(v2)
    v1 = v1.reshape(size, 1, -1)
    v2 = v2.reshape(1, size, -1)
    v1_min_v2 = v1 - v2
    return np.sqrt(np.einsum('ijx,ijx->ij', v1_min_v2, v1_min_v2))

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


def get_score_by_ICS(dict1, label1, dict2, label2):
    def get_ics_by_random(data, label1, adv_data, label2, batch_size):
        batch_num = int(np.ceil(adv_data.shape[0] / float(batch_size)))
        feature = None
        for i_batch in range(batch_num):
            start = i_batch * batch_size
            end = np.minimum(len(adv_data), (i_batch + 1) * batch_size)
            n_feed = end - start
            X_act = data[0:n_feed]
            label1 = label1[0:n_feed]
            # X_mean = np.mean(X_act, 0)
            X_adv_act = adv_data[start:end]
            label2 = label2[start:end]
            lid_batch_adv = cos_sim_in_class_global(X_act, label1, X_adv_act, label2)
            if feature is not None:

                feature = np.concatenate((feature, lid_batch_adv), 0)
            else:

                feature = lid_batch_adv
        return feature
    # assert len(dict1) == len(dict2)
    # print(dict1[0][0])
    # print('####################################')
    # print(dict2[0][0])
    # print('####################################')
    dim = len(dict1)
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

        adv_score[:, i] = get_ics_by_random(feature1, label1, feature2, label2, batch_size=1000)
    return adv_score

def get_train_model(args,train_features,labels,nb_layer,device):
    train_dir = 'detect_train_batchnorm/layers/'
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir,exist_ok=True)
    for l_indx in range(0,nb_layer):
        full_connect_model_path = '{}{}_{}.pth'.format(train_dir, args.dataset, l_indx)
        layer_data_path = '{}{}_{}_normal.npy'.format(train_dir,args.dataset,l_indx)
        if not os.path.isfile(layer_data_path):
            l_out = train_features[l_indx]
            np.save(layer_data_path,l_out)
        else:
            l_out = np.load(layer_data_path,allow_pickle=True)
        current_model = dense(input_shape=len(l_out[0]))
        print(l_out.shape)
        print(labels.shape)
        train(current_model,l_out,labels,args)
        torch.save(current_model.state_dict(),full_connect_model_path)

def get_train_features(args,features,labels,nb_layer,device,rand=0):
    dir_path = 'detect_train_batchnorm/layers/'
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
    lid = None
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
        if lid is not None:
            # lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            # lid = lid_batch
            lid_adv = lid_batch_adv
    return lid_adv


def get_score_by_IED(base_features, base_labels, test_features, test_labels):
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
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, -k:]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

def get_lacs_by_random(base, test, batch_size, k):
    batch_num = int(np.ceil(test.shape[0] / float(batch_size)))
    lid = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(test), (i_batch + 1) * batch_size)
        n_feed = end - start
        X_act = base[0:n_feed]
        X_mean = np.mean(base, 0)
        X_test = test[start:end]
        lid_batch = cos_sim(X_act - X_mean, X_test - X_mean, k)

        if lid is not None:
            lid = np.concatenate((lid, lid_batch), 0)

        else:
            lid = lid_batch
    return lid


def get_score_by_LACS(base_features, test_features):
    dim = len(base_features)
    test_score = np.zeros((len(test_features[0]), dim))

    for i in range(dim):
        if not isinstance(base_features[i],np.ndarray):
            feature1 = base_features[i].detach().numpy()
        else:
            feature1 = base_features[i]
        # print('feature1 is :', feature1[0])
        if not isinstance(test_features[i],np.ndarray):
            feature2 = test_features[i].detach().numpy()
        # print('feature2 is :', feature2[0])
        else:
            feature2 = test_features[i]

        test_score[:, i] = get_lacs_by_random(feature1, feature2, batch_size=100, k=10)
    return test_score


def cos_sim_global(data, batch, k):
    """
    input: shape is (n,m) n is batch_size,m is dimension
    output:shape is (n,n)
    """
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: np.mean(v)
    a = get_cos_similar_matrix(data, batch)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a


def get_acs_by_random(base, test, batch_size, k):
    batch_num = int(np.ceil(test.shape[0] / float(batch_size)))
    lid = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(test), (i_batch + 1) * batch_size)
        n_feed = end - start
        X_act = base[0:n_feed]
        X_mean = np.mean(X_act, 0)
        X_test = test[start:end]
        lid_batch = cos_sim_global(X_act - X_mean, X_test - X_mean, k)

        if lid is not None:
            lid = np.concatenate((lid, lid_batch), 0)

        else:
            lid = lid_batch

    return lid


def get_score_by_ACS(base_features, test_features):
    dim = len(base_features)
    test_score = np.zeros((len(test_features[0]), dim))

    for i in range(dim):
        if not isinstance(base_features[i], np.ndarray):
            feature1 = base_features[i].detach().numpy()
        else:
            feature1 = base_features[i]
        # print('feature1 is :', feature1[0])
        if not isinstance(test_features[i], np.ndarray):
            feature2 = test_features[i].detach().numpy()
        # print('feature2 is :', feature2[0])
        else:
            feature2 = test_features[i]

        test_score[:, i] = get_acs_by_random(feature1, feature2, batch_size=100, k=10)

    return test_score





def main():
    import argparse
    # from sklearn.manifold import TSNE
    parse = argparse.ArgumentParser(description='this is an adversarial detection based on Local Cosine Similarity')
    parse.add_argument('--seed', type=int, default=100, help='set the random seed')
    parse.add_argument('--gpu_index', type=str, default='1', help='the index of gpu')
    parse.add_argument('--dataset', default='mnist', choices=['MNIST', 'SVHN', 'CIFAR10','IMAGENET10'])
    parse.add_argument('--train',type=bool,default=False)
    parse.add_argument('--epochs',default=30)
    parse.add_argument('--batch_size',default=128)
    parse.add_argument('--type', default='iacs', choices=['knn', 'lacs', 'acs','ics', 'kd', 'lid', 'iacs', 'ilacs','ied'])
    parse.add_argument('--attack', default='fgsm', choices=['FGSM', 'PGD', 'DEEPFOOL', 'JSMA', 'CW2','CWL2','CWL0','CWLINF'])
    parse.add_argument('--test_attack', default='fgsm', choices=['FGSM', 'PGD', 'DEEPFOOL', 'JSMA', 'CW2','CWL2','CWL0','CWLINF'])
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
        from RawModels.MNISTConv import MNISTConvNet, MNIST_Training_Parameters, None_feature, get_feature
        from RawModels.Utils.dataset import get_mnist_test_loader,get_mnist_train_validate_loader
        rawModel_path = 'RawModels/MNIST/model/MNIST_raw.pt'

        test_loader = get_mnist_test_loader(dir_name='./RawModels/MNIST/',
                                            batch_size=MNIST_Training_Parameters['batch_size'])
        train_loader, _, _ = get_mnist_train_validate_loader(dir_name='./RawModels/MNIST/',
                                                             batch_size=MNIST_Training_Parameters['batch_size'])
        raw_model = MNISTConvNet().to(device)
        raw_model.load(path=rawModel_path, device=device)
        raw_model = PyTorchClassifier(raw_model,loss=None,input_shape=(1,28,28),nb_classes=10)
        nb_layer = 6
        rand = 1000

    if args.dataset == 'SVHN':
        from RawModels.SVHNConv import SVHNConvNet, SVHN_Training_Parameters, None_feature, get_feature
        from RawModels.Utils.dataset import get_svhn_test_loader,get_svhn_train_validate_loader
        rawModel_path = 'RawModels/SVHN/model/SVHN_raw.pt'
        test_loader = get_svhn_test_loader(dir_name='./RawModels/SVHN/',
                                           batch_size=SVHN_Training_Parameters['batch_size'])
        train_loader, _, _ = get_svhn_train_validate_loader(dir_name='./RawModels/SVHN/',
                                                            batch_size=SVHN_Training_Parameters['batch_size'])
        raw_model = SVHNConvNet().to(device)
        raw_model.load(path=rawModel_path, device=device)
        raw_model = PyTorchClassifier(raw_model,loss=None,input_shape=(3,32,32),nb_classes=10)
        nb_layer = 6
        rand = 1000
    if args.dataset == 'CIFAR10':
        from RawModels.ResNet import resnet20_cifar, CIFAR10_Training_Parameters, None_feature, get_feature
        from RawModels.Utils.dataset import get_cifar10_test_loader,get_cifar10_train_validate_loader
        rawModel_path = 'RawModels/CIFAR10/model/CIFAR10_raw.pt'
        test_loader = get_cifar10_test_loader(dir_name='./RawModels/CIFAR10/',
                                              batch_size=CIFAR10_Training_Parameters['batch_size'])
        train_loader, _, _ = get_cifar10_train_validate_loader(dir_name='./RawModels/CIFAR10/',
                                                               batch_size=CIFAR10_Training_Parameters['batch_size'])
        raw_model = resnet20_cifar().to(device)
        raw_model.load(path=rawModel_path, device=device)
        raw_model = PyTorchClassifier(raw_model,loss=None,input_shape=(3,32,32),nb_classes=10)
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
        nb_layer =9
        rand = 400



    if not os.path.exists(rawModel_path):
        print('please train model first!')
    # attack_type = args.attack
    dataset_type = args.dataset
    detect_type = args.type
    attack_list = ['FGSM', 'PGD', 'DEEPFOOL', 'JSMA', 'CWL0', 'CWL2', 'CWLINF']
    # test_attack_list = ['FGSM','PGD','DEEPFOOL','JSMA','CWL0','CWL2','CWLINF']

    # adv_load = adv_loader(attack, dataset_type, 32)
    test_feature_path = './detect_train/layers_features/{}_test.npy'.format(dataset_type)
    # adv_feature_path = './detect_train/layers_features/{}_{}.npy'.format(dataset_type, attack)
    base_feature_path = './detect_train/layers_features/{}_base.npy'.format(dataset_type)
    train_feature_path = './detect_train/layers_features/{}_train.npy'.format(dataset_type)
    if os.path.isfile(test_feature_path):
        test_features, test_labels = np.load(test_feature_path, allow_pickle=True)
    else:
        os.makedirs('./detect_train/layers_features/', exist_ok=True)
        test_features, test_labels = get_layers_feature(raw_model, test_loader, nb_layer, device, rand=rand)
        np.save(test_feature_path, [test_features, test_labels])
    # if os.path.isfile(adv_feature_path):
    #     adv_features, adv_labels = np.load(adv_feature_path, allow_pickle=True)
    #
    # else:
    #     adv_features, adv_labels = get_layers_feature(raw_model, adv_load, nb_layer, device, rand=0)
    #     np.save(adv_feature_path, [adv_features, adv_labels])
    if os.path.isfile(base_feature_path):
        base_features, base_labels = np.load(base_feature_path, allow_pickle=True)
    else:
        base_features, base_labels = get_layers_feature(raw_model, train_loader, nb_layer, device, rand=1000)
        np.save(base_feature_path, [base_features, base_labels])
    if os.path.isfile(train_feature_path):
        train_features, train_labels = np.load(train_feature_path, allow_pickle=True)
    else:

        train_features, train_labels = get_layers_feature(raw_model, train_loader, nb_layer, device, rand=0)
        np.save(train_feature_path, [train_features, train_labels])
    print(len(train_features[0]))
    print(len(train_labels))
    for attack in attack_list:
        adv_load = adv_loader(attack, dataset_type, 32)
        # test_feature_path = './detect_train/layers_features/{}_test.npy'.format(dataset_type)
        adv_feature_path = './detect_train/layers_features/{}_{}.npy'.format(dataset_type,attack)
        # base_feature_path = './detect_train/layers_features/{}_base.npy'.format(dataset_type)
        # train_feature_path = './detect_train/layers_features/{}_train.npy'.format(dataset_type)
        # if os.path.isfile(test_feature_path):
        #     test_features,test_labels = np.load(test_feature_path,allow_pickle=True)
        # else:
        #     os.makedirs('./detect_train/layers_features/',exist_ok=True)
        #     test_features,test_labels = get_layers_feature(raw_model,test_loader,nb_layer,device,rand=rand)
        #     np.save(test_feature_path,[test_features,test_labels])
        if os.path.isfile(adv_feature_path):
            adv_features,adv_labels = np.load(adv_feature_path,allow_pickle=True)

        else:
            adv_features,adv_labels = get_layers_feature(raw_model,adv_load,nb_layer,device,rand=0)
            np.save(adv_feature_path, [adv_features, adv_labels])
        # if os.path.isfile(base_feature_path):
        #     base_features,base_labels = np.load(base_feature_path,allow_pickle=True)
        # else:
        #     base_features,base_labels = get_layers_feature(raw_model,train_loader,nb_layer,device,rand=1000)
        #     np.save(base_feature_path,[base_features,base_labels])
        # if os.path.isfile(train_feature_path):
        #     train_features,train_labels = np.load(train_feature_path,allow_pickle=True)
        # else:
        #
        #     train_features,train_labels = get_layers_feature(raw_model,train_loader,nb_layer,device,rand=0)
        #     np.save(train_feature_path,[train_features,train_labels])
        # print(len(train_features[0]))
        # print(len(train_labels))
        # if args.train:
        #     get_train_model(args,train_features,train_labels,nb_layer,device)
        # print('base_features:',len(test_features[1][0]))
        # test_features,test_labels = get_train_features(args,test_features,test_labels,nb_layer,device)
        # adv_features, adv_labels = get_train_features(args,adv_features,adv_labels,nb_layer,device)
        # base_features,base_labels = get_train_features(args,base_features,base_labels,nb_layer,device)
        # train_features, train_labels = get_train_features(args,train_features,train_labels,nb_layer,device)

        # for test_attack in test_attack_list:
        args.attack = attack
        args.test_attack = attack
        feature_path = 'detect_without_train/{}_{}_{}.npy'.format(args.dataset, args.attack, args.type)
        # print('base_features:',len(test_features[1][0]))
        # print('test_features shape',test_features[0].shape)
        # print('adv_features shape',adv_features[0].shape)
        if not os.path.isfile(feature_path):
            if detect_type == 'ilacs':
                adv_score = get_score_by_ILACS(base_features, base_labels, adv_features, adv_labels)
                test_score = get_score_by_ILACS(base_features, base_labels, test_features, test_labels)
                # np.save('./Feature/' + attack_type + '.npy', [test_score, adv_score])
            if detect_type == 'lacs':
                test_score = get_score_by_LACS(base_features, test_features)
                adv_score = get_score_by_LACS(base_features, adv_features)
            elif detect_type == 'acs':
                test_score = get_score_by_ACS(base_features, test_features)
                adv_score = get_score_by_ACS(base_features, adv_features)
            elif detect_type == 'knn':
                test_score = get_score_by_KNN(base_features, test_features)
                adv_score = get_score_by_KNN(base_features, adv_features)
            elif detect_type == 'iacs':
                adv_score = get_score_by_IACS(base_features, base_labels, adv_features, adv_labels)
                test_score = get_score_by_IACS(base_features, base_labels, test_features, test_labels)
                # print('adv score :', adv_score)
            elif detect_type == 'ied':
                adv_score = get_score_by_IED(base_features, base_labels, adv_features, adv_labels)
                test_score = get_score_by_IED(base_features, base_labels, test_features, test_labels)
                # print('adv score :', adv_score)
            elif detect_type == 'ics':
                test_score = get_score_by_ICS(base_features, base_labels, test_features, test_labels)
                adv_score = get_score_by_ICS(base_features, base_labels, adv_features, adv_labels)

            else:
                print('input right detect type')
                exit(0)
            np.save(feature_path, [test_score, adv_score])
            # test_score, adv_score = tsne_(test_score),tsne_(adv_score)
        else:
            test_score, adv_score = np.load(feature_path,allow_pickle=True)
            # test_score,adv_score =  tsne_(test_score),tsne_(adv_score)
        # print('test score is :', test_score.mean(0))
        # print('adv score is :', adv_score.mean(0))
        # print('test_score shape:',test_score.shape)
        # print('adv_score shape:',adv_score.shape)
        mix_data, mix_label = mix_feature(test_score, adv_score)

        test_attack = args.attack
        if attack != test_attack:
            path = './detect_without_train/{}_{}_{}.npy'.format(args.dataset, args.attack, args.type)
            if os.path.exists(path):
                test_score, adv_score = np.load(path,allow_pickle=True)
            else:
                print('first generate the feature of attack!')
                exit()

            mix_data, mix_label = mix_feature(test_score, adv_score)
            # print(mix_data.shape)
            scale = MinMaxScaler().fit(mix_data)
            mix_data = scale.transform(mix_data)
            x_train, _, y_train, _ = train_test_split(mix_data, mix_label, random_state=0, test_size=0.33)
            path1 = './detect_train_without/{}_{}_{}.npy'.format(args.dataset,args.test_attack,args.type)
            test_score1, adv_score1 = np.load(path1,allow_pickle=True)
            mix_data1, mix_label1 = mix_feature(test_score1, adv_score1)
            _, x_test, _, y_test = train_test_split(mix_data1, mix_label1, random_state=0, test_size=0.33)
        else:
            # path = './Feature/' + detect_type + '_' + attack_type + '.npy'
            # test_score, adv_score = np.load(path)
            mix_data, mix_label = mix_feature(test_score, adv_score)
            x_train, x_test, y_train, y_test = train_test_split(mix_data, mix_label, random_state=0, test_size=0.33)
        # tsne = TSNE(n_components=2)
        # tsne(mix_data,mix_label)
        # mix_data, mix_label = tsne(mix_data, mix_label)
        x_train_idx = np.isnan(x_train)
        x_test_idx = np.isnan(x_test)
        x_train[x_train_idx] = 0
        x_test[x_test_idx] = 0
        # print('nan:', np.isnan(x_train).any())
        # print('inf:', np.isinf(x_train).any())
        lr = LogisticRegressionCV(max_iter=1000).fit(x_train, y_train)
        # print(x_test.shape)
        predict_score = lr.predict_proba(x_test)[:, 1]
        plot_auc(y_test, predict_score, attack,test_attack, dataset_type, detect_type)


if __name__=="__main__":
    main()


