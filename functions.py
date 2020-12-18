import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import tensorflow.keras.backend as K
import os
import pickle
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import roc_curve 
import statistics as stat

def make_opts():
    opts = {'lr' : random.choice(np.arange(0.0001, 0.00101, 0.00005)), 'mx_epochs' : random.choice(np.arange(40, 201, 20)), 'val_freq' : random.choice(np.arange(10, 81, 10)).item(), 'F1S' : random.choice(np.asarray([3, 5, 10, 15, 20, 25, 30, 32, 35, 40])), 
            'F1N' : random.choice(np.asarray([10, 20, 30, 40])), 'F2S' : random.choice(np.asarray([3, 5, 10, 15, 20, 25])), 'F2N' : random.choice(np.asarray([10, 20, 30, 40])), 'F3S' : random.choice(np.asarray([3, 5, 10, 15])), 'F3N' : random.choice(np.asarray([10, 20, 30, 40])), 'batch_size' : random.choice(np.asarray([8, 16, 32, 64]))}
    return opts

def show_slice(image, subject_type, jj):
    plt.title('{0:s} slice {1:d}'.format(subject_type, jj))
    plt.imshow(image, cmap="gray")
    return plt

def cnn_network(opts):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(opts['siz'], opts['siz'], 1), name = 'input'))
    model.add(layers.Conv2D(opts['F1N'], (opts['F1S'], opts['F1S']), padding = 'same', name = 'conv_1', activation = 'relu'))
    model.add(layers.BatchNormalization(name = 'BN_1'))
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(layers.Conv2D(opts['F2N'], (opts['F2S'], opts['F2S']), strides = 2, padding = 'same', name = 'conv_2', activation = 'relu'))
    model.add(layers.BatchNormalization(name = 'BN_2'))
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(layers.Conv2D(opts['F3N'], (opts['F3S'], opts['F3S']), padding = 'same', name = 'conv_3', activation = 'relu'))
    model.add(layers.BatchNormalization(name = 'BN_3'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', name = 'dense_1'))
    model.add(layers.Dense(2, name = 'dense_2', activation = 'softmax'))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=opts['lr'], momentum = 0.9), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])  
    return model

def split(X, y, num_fold):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80, stratify = y)
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    T = {'xtest': X_test, 'ytest': y_test}
    S = {'xtrain' : [], 'ytrain' : [], 'xval' : [], 'yval' : []}
    kf = StratifiedKFold(n_splits=num_fold, shuffle = True)
    for train_index, test_index in kf.split(X_train, y_train):
        S['xtrain'].append(X_train[train_index]) 
        S['xval'].append(X_train[test_index])
        S['ytrain'].append(y_train[train_index])
        S['yval'].append(y_train[test_index])
    return T, S

def activation_weights(net, im, list_layers):
    H = [None] * len(list_layers)
    for i in range(0, len(list_layers)):
        q = im
        q = np.expand_dims(q, axis=0)
        activations = K.function([net.layers[0].input], net.get_layer(name = list_layers[i]).output)
        act1 = activations([q, 0])
        imgSize = im.shape[0:2]
        _, _, _, maxValueIndex = np.where(act1 >= act1.max())
        act1chMax = act1[:, :, :, maxValueIndex[0]]
        act1chMax = (act1chMax - act1chMax.min()) / act1chMax.max()
        H[i] = act1chMax
    return H
 
def process_results(opts_dir, net_dir): 
    files = sorted(os.scandir(opts_dir), key=lambda e: e.name)
    files2 = sorted(os.scandir(net_dir), key=lambda e: e.name) 
    C = {'ac' : np.zeros((1000), dtype = float), 'ppv' : np.zeros((1000), dtype = float), 'npv' : np.zeros((1000), dtype = float), 'cm' : np.zeros((1000, 2, 2), dtype = int),
        'spc' : np.zeros((1000), dtype = float), 'sen' : np.zeros((1000), dtype = float), 'auc' : np.zeros((1000), dtype = float),
        'ax' : [None] * 1000, 'ay' : [None] * 1000, 'cnt' : 0}
    A = [[]] * 3
    metric_stats = []
    list_layers = ['conv_1', 'conv_2', 'conv_3']
    
    for i in range(0, len(files)):
        if os.path.isfile(files[i]):
            f = open(files[i], "rb")
            d = pickle.load(f) 
            f.close()
            
            tt = d['T']['ytest'].flatten() 
            
            for j in d['ypred']:
                pp = j.flatten() 
                cm = tf.math.confusion_matrix(tt, pp).numpy()
                ax, ay, T = roc_curve(tt, pp)
                auc = roc_auc_score(tt, pp)
                
                C['cm'][C['cnt']] = cm
                C['ac'][C['cnt']] = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
                C['ppv'][C['cnt']] = cm[0, 0] / np.sum(cm[0, :])
                C['npv'][C['cnt']] = cm[1, 1] / np.sum(cm[1, :])
                C['sen'][C['cnt']] = cm[0, 0] / np.sum(cm[:, 0])
                C['spc'][C['cnt']] = cm[1, 1] / np.sum(cm[:, 1])
                C['auc'][C['cnt']] = auc
                C['ax'][C['cnt']] = ax
                C['ay'][C['cnt']] = ay
                C['cnt'] = C['cnt'] + 1
            
            
            for j in os.scandir(files2[i]):
                net = tf.keras.models.load_model(j)
                for tt in d['T']['xtest']:
                    H = activation_weights(net, tt, list_layers)
                    for k in range(0, len(A)):
                        if (len(A[k]) == 0):
                            A[k] = H[k]
                        else: 
                            A[k] = A[k] + H[k]  
    
    for i in range(0, len(A)):
        A[i] = (A[i] - A[i].min()) / A[i].max() 
        
    metric_stats = {'ac_mean' : C['ac'].mean(axis=0), 'ac_std' : C['ac'].std(axis=0), 'ac_mean_minus_std' : C['ac'].mean(axis=0) - C['ac'].std(axis=0), 'ppv_mean' : C['ppv'].mean(axis=0), 'ppv_std' : C['ppv'].std(axis=0), 'ppv_mean_minus_std' : C['ppv'].mean(axis=0) - C['ppv'].std(axis=0), 'npv_mean' : C['npv'].mean(axis=0), 'npv_std' : C['npv'].std(axis=0), 'npv_mean_minus_std' : C['npv'].mean(axis=0) - C['npv'].std(axis=0), 'sen_mean' : C['sen'].mean(axis=0), 'sen_std' : C['sen'].std(axis=0), 'sen_mean_minus_std' : C['sen'].mean(axis=0) - C['sen'].std(axis=0), 'spc_mean' : C['spc'].mean(axis=0), 'spc_std' : C['spc'].std(axis=0), 'spc_mean_minus_std' : C['spc'].mean(axis=0) - C['spc'].std(axis=0), 'auc_mean' : C['auc'].mean(axis=0), 'auc_std' : C['auc'].std(axis=0), 'auc_mean_minus_std' : C['auc'].mean(axis=0) - C['auc'].std(axis=0)} 
    return C, A, metric_stats

def show_activations(A):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fix2, ax3 = plt.subplots()
    ax1.set_title('conv_1 activations')
    ax1.imshow(A[0][0, :, :], cmap="hot")
    ax2.set_title('conv_2 activations')
    ax2.imshow(A[1][0, :, :], cmap="hot")
    ax3.set_title('conv_3 activations')
    ax3.imshow(A[2][0, :, :], cmap="hot")
    return fig1, fig2, fig3

def plot_roc(C):
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(C['ax'], C['ay'], label='area = {:.3f}'.format(C['auc']))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    return plt
    
def grid_results(folder):
    P = []
    A = []
    F = []
    
    files = sorted(os.scandir(folder), key=lambda e: e.name)           
    for path in files:
        if os.path.isfile(path):
            f = open(path, "rb")
            opts = pickle.load(f) 
            f.close()
            p = [None] * 10
            fields = list(opts.keys())
            for j in range(0, 10):
                p[j] = opts[fields[j]]
            P.append(p)
            A.append(opts['acc'])
            F.append(f.name)
    D = {'P' : P, 'A' : np.asarray(A), 'F' : F}
    opts, opt_params_stats = optimal_grid_parameters(D)
    return D, opts, opt_params_stats

def optimal_grid_parameters(D):
    a = D['A'].mean(axis=1)
    b = D['A'].mean(axis=1) - D['A'].std(axis = 1)
    idx = np.argmax(b)
    opt_params_stats = {'max_mean_acc' : a.max(), 'max_ma_D_idx' : np.argmax(a), 'max_mean_acc_min_std' : b.max(), 'max_mams_D_idx' : idx}
    P = D['P'][idx]
    opts = {'lr' : P[0], 'mx_epochs' : P[1], 'val_freq' : P[2], 'F1S' : P[3], 
            'F1N' : P[4], 'F2S' : P[5], 'F2N' : P[6], 'F3S' : P[7], 'F3N' : P[8], 'batch_size' : P[9]}
    return opts, opt_params_stats

def process_slice(opts_dir, net_dir):
    files = sorted(os.scandir(opts_dir), key=lambda e: e.name) 
    files2 = sorted(os.scandir(net_dir), key=lambda e: e.name) 
    C = []
    metric_stats = []
    cnt = 0
    pslice = [0] * 157        
    list_layers = ['conv_1', 'conv_2', 'conv_3']
    
    for i in range(0, len(files)):
        if os.path.isfile(files[i]):
            f = open(files[i], "rb")
            d = pickle.load(f) 
            f.close()
            A = [[]] * 3
            tt = d['T']['ytest'].flatten() 
            C.append({'slice' : d['jj'], 'cm' : np.zeros((2, 2), dtype = int),
                      'ac' : np.zeros(len(d['ypred']), dtype = float), 'ppv' : np.zeros(len(d['ypred']), dtype = float),
                      'npv' : np.zeros(len(d['ypred']), dtype = float), 'A' : []})
            for j in range(0, len(d['ypred'])):
                pp = d['ypred'][j].flatten() 
                cm = tf.math.confusion_matrix(tt, pp).numpy()
                
                C[cnt]['cm'] = C[cnt]['cm'] + cm
                C[cnt]['ac'][j] = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
                C[cnt]['ppv'][j] = cm[0, 0] / np.sum(cm[0, :])
                C[cnt]['npv'][j] = cm[1, 1] / np.sum(cm[1, :])
            
            metric_stats.append({'ac_mean' : C[cnt]['ac'].mean(axis=0), 'ac_std' : C[cnt]['ac'].std(axis=0), 'ac_mean_minus_std' : C[cnt]['ac'].mean(axis=0) - C[cnt]['ac'].std(axis=0), 'ppv_mean' : C[cnt]['ppv'].mean(axis=0), 'ppv_std' : C[cnt]['ppv'].std(axis=0), 'ppv_mean_minus_std' : C[cnt]['ppv'].mean(axis=0) - C[cnt]['ppv'].std(axis=0), 'npv_mean' : C[cnt]['npv'].mean(axis=0), 'npv_std' : C[cnt]['npv'].std(axis=0), 'npv_mean_minus_std' : C[cnt]['npv'].mean(axis=0) - C[cnt]['npv'].std(axis=0)})
                            
            for j in os.scandir(files2[i]):
                net = tf.keras.models.load_model(j)
                H = activation_weights(net, d['T']['xtest'][0], list_layers)
                for k in range(0, len(A)):
                    if (len(A[k]) == 0):
                        A[k] = H[k]
                    else: 
                        A[k] = A[k] + H[k]  
            
            for m in range(0, len(A)):
                A[m] = (A[m] - A[m].min()) / A[m].max() 
            
            pslice[C[cnt]['slice']] = 1
            C[cnt]['A'] = A
            cnt = cnt + 1
    return C, pslice, metric_stats

def slice_results(folder):
    P = []
    A = []
    F = []
    S = []
    
    files = sorted(os.scandir(folder), key=lambda e: e.name)           
    for path in files:
        if os.path.isfile(path):
            f = open(path, "rb")
            opts = pickle.load(f) 
            f.close()
            p = [None] * 10
            fields = list(opts.keys())
            for j in range(0, 10):
                p[j] = opts[fields[j]]
            P.append(p)
            A.append(opts['acc'])
            F.append(f.name)
            S.append(opts['jj'])
    D = {'P' : P, 'A' : np.asarray(A), 'F' : F, 'S' : S}
    opts, opt_slice_stats = optimal_slice(D)
    return D, opts, opt_slice_stats

def optimal_slice(D):
    a = D['A'].mean(axis=1)
    b = D['A'].mean(axis=1) - D['A'].std(axis = 1)
    idx = np.argmax(b)
    opt_slice_stats = {'max_mean_acc' : a.max(), 'max_ma_D_idx' : np.argmax(a), 'max_mean_acc_min_std' : b.max(), 'max_mams_D_idx' : idx}
    P = D['P'][idx]
    S = D['S'][idx]
    opts = {'lr' : P[0], 'mx_epochs' : P[1], 'val_freq' : P[2], 'F1S' : P[3], 
            'F1N' : P[4], 'F2S' : P[5], 'F2N' : P[6], 'F3S' : P[7], 'F3N' : P[8], 'batch_size' : P[9], 'jj' : S, 'P' : './Data/P_left_sm.mat', 'C' : './Data/C_sm.mat', 'res_dir' : './run_opt_cv_opts'}
    return opts, opt_slice_stats