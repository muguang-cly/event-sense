import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tsai.all import *
from fastai.metrics import accuracy
from scipy import signal

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

        total_loss = classification_loss + regression_loss
        return total_loss



def classification_accuracy(inp, targ):
    class_preds = inp[:, :7]
    targ_classes = targ[:, :7].argmax(dim=1)
    return accuracy(class_preds, targ_classes)

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

final_labels = np.random.rand(1000, 31)
X=  np.random.rand(1000, 31)
splits = get_splits(final_labels, valid_size=0, stratify=True, random_state=24, shuffle=True)
dls = get_ts_dls(X, final_labels, splits=splits, bs=100)
model = InceptionTimePlus(
    c_in=2, bn=True, c_out=31,
    seq_len=1000, nf=16, concat_pool=True,
    fc_dropout=0, ks=60, coord=False, depth=9,
    separable=False, dilation=2, stride=1,
    sa=True, se=None, act=torch.nn.modules.activation.LeakyReLU)

model.eval()
learn = ts_learner(
    dls,
    model,
    loss_func=CustomLoss(),
    metrics=[classification_accuracy, force_mae, distance_mae]
)
learn.load('ISSN')

file_path = r'test_data.csv'
df = pd.read_csv(file_path, header=None)

pre_cate = []
name = []
force_pred1 = []
force_pred2 = []
distance_pred1 = []
distance_pred2 = []
bad_file = []
force1_targets = []
force2_targets = []
distance1_targets = []
distance2_targets = []

for i in range(1000):
    a = df.iloc[i, 0]
    name.append(a)
    force_targets1 = df.iloc[i, 1].astype(float)
    force_targets2 = df.iloc[i, 2].astype(float)
    distance_targets1 = df.iloc[i, 3].astype(float)
    distance_targets2 = df.iloc[i, 4].astype(float)
    force1_targets.append(force_targets1)
    force2_targets.append(force_targets2)
    distance1_targets.append(distance_targets1)
    distance2_targets.append(distance_targets2)

    curve_distance = df.iloc[i, 5:3005].values.astype(float)

    curve_force = df.iloc[i, 3005:6005].values.astype(float)

    curve_distance = np.trim_zeros(curve_distance, 'b')

    curve_force = np.trim_zeros(curve_force, 'b')


    def linear_interpolation(y, new_length):
        original_length = len(y)
        x_original = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, new_length)
        y_new = np.interp(x_new, x_original, y)
        return y_new


    data_len = 610
    force_interpolation = signal.medfilt(signal.medfilt(linear_interpolation(curve_force, data_len), kernel_size=3),
                                         kernel_size=5)
    curve_interpolation = signal.medfilt(signal.medfilt(linear_interpolation(curve_distance, data_len), kernel_size=3),
                                         kernel_size=5)


    def fill_array_with_zeros(array, target_length):
        if len(array) >= target_length:
            return array
        else:
            padding_length = target_length - len(array)
            padded_array = np.pad(array, (0, padding_length), mode='constant')
            return padded_array


    force_padding = fill_array_with_zeros(force_interpolation, 1000)
    distance_padding = fill_array_with_zeros(curve_interpolation, 1000)

    force_max = 33
    force_min = 0
    distance_max = 295
    distance_min = 0

    force_padding_normal = (force_padding - force_min) / (force_max - force_min) * 2 - 1
    distance_padding_normal = (distance_padding - distance_min) / (distance_max - distance_min) * 2 - 1

    force_padding_normal = force_padding_normal.reshape(1, 1, 1000)
    distance_padding_normal = distance_padding_normal.reshape(1, 1, 1000)

    raw_t = np.concatenate((force_padding_normal, distance_padding_normal), axis=1)

    raw_y = torch.randn((1, 31))

    probas, targets, preds = learn.get_X_preds(raw_t, raw_y)
    preds = np.array(preds)

    class_preds = preds[:, :7]
    reg_preds = preds[:, 7:]

    class_preds_tensor = torch.tensor(class_preds)
    probabilities = F.softmax(class_preds_tensor, dim=1)

    predicted_classes = np.array(torch.max(probabilities, 1)[1])
    fold_num_pred = predicted_classes.item()

    force_preds = (reg_preds[:, 0:12] + 1) / 2 * (force_max - force_min) + force_min
    distance_preds = (reg_preds[:, 12:24] + 1) / 2 * (distance_max - distance_min) + distance_min
    pre_cate.append(fold_num_pred)
    if fold_num_pred == 1:
        name.append(a)
        force_preds_fold = force_preds[0][0:2 * int(fold_num_pred)]
        distance_preds_fold = distance_preds[0][0:2 * int(fold_num_pred)]
        force_pred1.append(force_preds_fold[0])
        force_pred2.append(force_preds_fold[1])

        distance_pred1.append(distance_preds_fold[0])
        distance_pred2.append(distance_preds_fold[1])

        force_numpy = (np.array(raw_t[:, 0, :][0]) + 1) / 2 * (force_max - force_min) + force_min
        distance_numpy = (np.array(raw_t[:, 1, :][0]) + 1) / 2 * (distance_max - distance_min) + distance_min



    else:
        bd = 0
        force_pred1.append(bd)
        force_pred2.append(bd)
        distance_pred1.append(bd)
        distance_pred2.append(bd)

new_df = df.copy()
new_df['pre_cate'] = pre_cate
new_df['force_pred1'] = force_pred1
new_df['force_pred2'] = force_pred2
new_df['distance_pred1'] = distance_pred1
new_df['distance_pred2'] = distance_pred2

new_file_path = r'save_testdata.csv'
new_df.to_csv(new_file_path, index=False, header=False)

