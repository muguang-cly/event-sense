import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.optim
from tsai.all import *
from fastai.callback.tracker import SaveModelCallback
from sklearn.preprocessing import StandardScaler
from fastai.callback.tracker import CSVLogger
from fastai.callback.tracker import SaveModelCallback, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from fastai.metrics import accuracy


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, preds, target):
        preds_classification = preds[:, :7]
        preds_regression = preds[:, 7:]

        target_classification = target[:, :7].argmax(dim=1)
        target_regression = target[:, 7:]

        classification_loss = self.cross_entropy(preds_classification, target_classification)

        mask = (target_regression != -10.0).float()
        preds_regression = preds_regression * mask
        target_regression = target_regression * mask

        regression_loss = self.mse(preds_regression, target_regression)

        total_loss = 0.1 * classification_loss + 2 * regression_loss
        return total_loss

class CustomLoss_nomask(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, preds, target):
        preds_classification = preds[:, :7]
        preds_regression = preds[:, 7:]

        target_classification = target[:, :7].argmax(dim=1)
        target_regression = target[:, 7:]

        classification_loss = self.cross_entropy(preds_classification, target_classification)

        regression_loss = self.mse(preds_regression, target_regression)

        total_loss = classification_loss + regression_loss
        return total_loss


class CustomLossfd(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.classification_weight = 0.1
        self.force_weight = 1.0
        self.distance_weight = 1.0

    def forward(self, preds, target):
        preds_classification = preds[:, :7]
        preds_regression = preds[:, 7:]

        target_classification = target[:, :7].argmax(dim=1)
        target_regression = target[:, 7:]

        classification_loss = self.cross_entropy(preds_classification, target_classification)

        mask = (target_regression != -10.0).float()

        preds_regression_masked = preds_regression * mask
        target_regression_masked = target_regression * mask

        preds_force = preds_regression_masked[:, :12]
        preds_distance = preds_regression_masked[:, 12:]
        target_force = target_regression_masked[:, :12]
        target_distance = target_regression_masked[:, 12:]

        force_loss = self.mse(preds_force, target_force)
        distance_loss = self.mse(preds_distance, target_distance)

        total_loss = (self.classification_weight * classification_loss +
                      self.force_weight * force_loss +
                      self.distance_weight * distance_loss)

        return total_loss


class CustomLossfd1(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.classification_weight = nn.Parameter(tensor([1.]))
        self.force_weight = nn.Parameter(tensor([1.]))
        self.distance_weight = nn.Parameter(tensor([1.]))

    def forward(self, preds, target):
        preds_classification = preds[:, :7]
        preds_regression = preds[:, 7:]

        target_classification = target[:, :7].argmax(dim=1)
        target_regression = target[:, 7:]

        classification_loss = self.cross_entropy(preds_classification, target_classification)

        mask = (target_regression != -10.0).float()

        preds_regression_masked = preds_regression * mask
        target_regression_masked = target_regression * mask

        preds_force = preds_regression_masked[:, :12]
        preds_distance = preds_regression_masked[:, 12:]
        target_force = target_regression_masked[:, :12]
        target_distance = target_regression_masked[:, 12:]

        force_loss = self.mse(preds_force, target_force)
        distance_loss = self.mse(preds_distance, target_distance)

        print(self.classification_weight)
        print(self.force_weight)
        print(self.distance_weight)

        total_loss = (self.classification_weight * classification_loss +
                      self.force_weight * force_loss +
                      self.distance_weight * distance_loss)

        return total_loss

def classification_accuracy(inp, targ):
    class_preds = inp[:, :7]
    targ_classes = targ[:, :7].argmax(dim=1)
    return accuracy(class_preds, targ_classes)

def regression_mae(inp, targ):
    reg_preds = inp[:, 7:]
    reg_targs = targ[:, 7:]
    mask = reg_targs != -10.0
    return torch.nn.functional.l1_loss(reg_preds[mask], reg_targs[mask], reduction='mean')


def force_mae(inp, targ, force_mean=10, force_std=4):
    force_preds = inp[:, 7:19]
    force_targs = targ[:, 7:19]

    force_preds_denorm = (force_preds + 1) / 2 * (force_max - force_min) + force_min
    force_targs_denorm = (force_targs + 1) / 2 * (force_max - force_min) + force_min

    mask = force_targs_denorm != (-10.0 + 1) / 2 * (force_max - force_min) + force_min
    return F.l1_loss(force_preds_denorm[mask], force_targs_denorm[mask], reduction='mean')


def distance_mae(inp, targ, distance_mean=200, distance_std=80):
    distance_preds = inp[:, 19:]
    distance_targs = targ[:, 19:]

    distance_preds_denorm = (distance_preds + 1) / 2 * (distance_max - distance_min) + distance_min
    distance_targs_denorm = (distance_targs + 1) / 2 * (distance_max - distance_min) + distance_min

    mask = distance_targs_denorm != (-10.0 + 1) / 2 * (distance_max - distance_min) + distance_min
    return F.l1_loss(distance_preds_denorm[mask], distance_targs_denorm[mask], reduction='mean')



df_force_0 = pd.DataFrame(np.load('force_ploy_noise_0_1000.npy', mmap_mode='r+')).iloc[0:5000]
df_distance_0 = pd.DataFrame(np.load('distance_ploy_noise_0_1000.npy', mmap_mode='r+')).iloc[0:5000]

df_force_1 = pd.DataFrame(np.load('force_ploy_noise_1_1000.npy', mmap_mode='r+')).iloc[0:5000]
df_distance_1 = pd.DataFrame(np.load('distance_ploy_noise_1_1000.npy', mmap_mode='r+')).iloc[0:5000]

df_force_2 = pd.DataFrame(np.load('force_ploy_noise_2_1000.npy', mmap_mode='r+')).iloc[0:5000]
df_distance_2 = pd.DataFrame(np.load('distance_ploy_noise_2_1000.npy', mmap_mode='r+')).iloc[0:5000]

df_force_3 = pd.DataFrame(np.load('force_ploy_noise_3_1000.npy', mmap_mode='r+')).iloc[0:5000]
df_distance_3 = pd.DataFrame(np.load('distance_ploy_noise_3_1000.npy', mmap_mode='r+')).iloc[0:5000]

df_force_4 = pd.DataFrame(np.load('force_ploy_noise_4_1000.npy', mmap_mode='r+')).iloc[0:5000]
df_distance_4 = pd.DataFrame(np.load('distance_ploy_noise_4_1000.npy', mmap_mode='r+')).iloc[0:5000]

df_force_5 = pd.DataFrame(np.load('force_ploy_noise_5_1000.npy', mmap_mode='r+')).iloc[0:5000]
df_distance_5 = pd.DataFrame(np.load('distance_ploy_noise_5_1000.npy', mmap_mode='r+')).iloc[0:5000]

df_force_6 = pd.DataFrame(np.load('force_ploy_noise_6_1000.npy', mmap_mode='r+')).iloc[0:5000]
df_distance_6 = pd.DataFrame(np.load('distance_ploy_noise_6_1000.npy', mmap_mode='r+')).iloc[0:5000]

df_force = pd.concat([df_force_0, df_force_1, df_force_2, df_force_3, df_force_4, df_force_5, df_force_6], axis=0)
df_distance = pd.concat(
    [df_distance_0, df_distance_1, df_distance_2, df_distance_3, df_distance_4, df_distance_5, df_distance_6], axis=0)

force_max = 33
force_min = 0
distance_max = 295
distance_min = 0

x_force = df_force.iloc[:, -1000:].values
x_distance = df_distance.iloc[:, -1000:].values


x_force_normalized = (x_force - force_min) / (force_max - force_min) * 2 - 1
x_distance_normalized = (x_distance - distance_min) / (distance_max - distance_min) * 2 - 1

X_force = np.reshape(x_force_normalized,(x_force_normalized.shape[0], 1, x_force_normalized.shape[1]))
X_distance = np.reshape(x_distance_normalized,(x_distance_normalized.shape[0], 1, x_distance_normalized.shape[1]))
X = np.concatenate((X_force, X_distance),axis=1)
X = X.astype(np.float32)

y_force_1 = df_force.iloc[:, 1:3].values
y_force_2 = df_force.iloc[:, 3:5].values
y_force_3 = df_force.iloc[:, 5:7].values
y_force_4 = df_force.iloc[:, 7:9].values
y_force_5 = df_force.iloc[:, 9:11].values
y_force_6 = df_force.iloc[:, 11:13].values

y_force = np.concatenate(
    (y_force_1, y_force_2, y_force_3, y_force_4, y_force_5, y_force_6), axis=1
)

y_distance_1 = df_distance.iloc[:, 1:3].values
y_distance_2 = df_distance.iloc[:, 3:5].values
y_distance_3 = df_distance.iloc[:, 5:7].values
y_distance_4 = df_distance.iloc[:, 7:9].values
y_distance_5 = df_distance.iloc[:, 9:11].values
y_distance_6 = df_distance.iloc[:, 11:13].values

y_distance = np.concatenate(
    (y_distance_1,y_distance_2,y_distance_3,y_distance_4,y_distance_5,y_distance_6), axis=1
)

y_force = np.where(y_force==0, 9735, y_force)
y_distance = np.where(y_distance==0, 9735, y_distance)

y_force_norm = (y_force - force_min) / (force_max - force_min) * 2 - 1
y_distance_norm = (y_distance - distance_min) / (distance_max - distance_min) * 2 - 1

y_1 = np.concatenate((y_force_norm, y_distance_norm),axis=1)

y_1_f = np.where(y_1==589.0, -10.0, y_1)
y_1_d = np.where(y_1_f==65.0, -10.0, y_1_f)
y_nums = np.expand_dims(df_force.iloc[:,0].values, axis=1)
y = np.concatenate((y_nums, y_1_d),axis=1)

n = len(y)
num_classes = 7

one_hot_labels = np.zeros((n, num_classes))

for idx, label in enumerate(y[:, 0].astype(int)):
    one_hot_labels[idx, label] = 1

final_labels = np.concatenate((one_hot_labels, y[:, 1:]), axis=1).astype(np.float32)


splits = get_splits(final_labels, valid_size=0.3, stratify=True, random_state=24, shuffle=True)
dls = get_ts_dls(X, final_labels, splits=splits,bs=100)


path='model_save_file'
epoch = 100
ISSN = InceptionTimePlus(
    c_in=dls.vars, bn=True, c_out=31,
    seq_len=dls.len, nf=16, concat_pool=True,
    fc_dropout=0, ks=60, coord=False, depth=9,
    separable=False, dilation=2, stride=1,
    sa=True, se=None, act=torch.nn.modules.activation.LeakyReLU)

ISSN.eval()
learn = ts_learner(
    dls,
    ISSN,
    loss_func=CustomLoss(),
    metrics=[classification_accuracy, force_mae, distance_mae],
    cbs=[SaveModelCallback(monitor='valid_loss', fname=path+'/'+'ISSN'),
         CSVLogger(fname=path+'/'+'ISSN.csv')]
)

learn.fit_one_cycle(epoch, 1e-4)



