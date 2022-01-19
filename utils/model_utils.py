# Loading libraries
import numpy as np
import pandas as pd
import os
import pickle
from collections import OrderedDict
from joblib import dump, load
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, dataloader


def split_data(df, label='LABELS', valid_size=0.05):
    x = df.drop(label, axis=1)
    y = df[label].values
    
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=valid_size, shuffle=True)

    train_idx = x_train.index.values
    val_idx = x_val.index.values

    return x, y, train_idx, val_idx



class Scaler:
    def __init__(self, x):
        self.mean = np.expand_dims(x.mean(axis=(0)), (0))
        self.std = np.expand_dims(x.std(axis=(0)), (0))
    def scale(self, x):  
        return ((x-self.mean)/self.std)


# Defines an lstm model architecture
class LstmNet(nn.Module):
    def __init__(self, feat_dim, hid_dim, num_layers, y_dim, lr, wd, n_epochs, batch_size, pos_weight, bidirectional, dropout, state_comb, n_steps):
        super(LstmNet, self).__init__()
        self.lstm = nn.LSTM(feat_dim, hid_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        lin_hid_dim = hid_dim
        if bidirectional:
            lin_hid_dim = lin_hid_dim*2
        if state_comb == 'all':
            lin_hid_dim = lin_hid_dim*n_steps
        self.linear_hid = nn.Linear(lin_hid_dim, hid_dim)
        self.linear_out = nn.Linear(hid_dim, y_dim)
        self.lin_dropout = nn.Dropout(dropout)
        self.lr = lr
        self.wd = wd
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.pos_weight = pos_weight
        self.state_comb = state_comb
    
    def forward(self, x):
        h, _ = self.lstm(x)
        if self.state_comb == 'last':
            x = self.linear_hid(h[:,-1,:])
        elif self.state_comb == 'avg':
            x = self.linear_hid(h.mean(dim=1))
        elif self.state_comb == 'all':
            x = self.linear_hid(h.flatten(start_dim=1))
        else:
            exit()
        x = self.lin_dropout(x)
        out = self.linear_out(x)
        return out

    def predict_proba(self, x):
        out = F.softmax(self.forward(x)).numpy()
        return out
    
    def predict(self, x, thres=.5):
        out = self.predict_proba(x)
        preds = np.argmax(out, axis=1)
        return preds


# defining meta model class
class Modeler:
    def __init__(self, algo_name, algo_class, params, models_path):
        self.model = algo_class(**params)
        self.model_name = algo_name
        self.params = params
        self.models_path = models_path
        
    def train(self, x_train, x_val, y_train, y_val):
        if self.model_name.startswith('nn_'):
            self.nn_fit(x_train, x_val, y_train, y_val)
            #scores = self.score(x_train, y_train)
            self.model_file = os.path.join(self.models_path, f'{self.model_name}.pt')
            torch.save(self.model.state_dict(), self.model_file)
        else:
            x_train = x_train.reshape(x_train.shape[0],-1)
            self.model.fit(x_train, y_train)
            #scores = self.score(x_train, y_train)
            self.model_file = os.path.join(self.models_path, f'{self.model_name}.joblib')
            dump(self.model, self.model_file) 
        #return scores

    def predict_proba(self, x):
        if not self.model_name.startswith('nn_'):
            if x.ndim == 3:
                x = x.reshape(x.shape[0],-1)
            pred = self.model.predict_proba(x)[:,1]
        else:
            x = torch.tensor(x.astype(np.float32))
            with torch.no_grad():
               pred = self.model.predict_proba(x) 
        return pred

    def predict(self, x):
        if not self.model_name.startswith('nn_') and x.ndim == 3:
            x = x.reshape(x.shape[0],-1)
            pred = self.model.predict(x)
        else:
            x = torch.tensor(x.astype(np.float32))
            with torch.no_grad():
               pred = self.model.predict(x) 
        return pred

    def score(self, x, y):
        y_pred = self.predict(x)
        y_pred_probas = self.predict_proba(x)
        acc = round(accuracy_score(y, y_pred), 5)
        #f1 = round(f1_score(y, y_pred), 5)
        #fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_probas)
        #auc = round(metrics.auc(fpr,tpr), 5)
        conf_mat = confusion_matrix(y, y_pred)
        #scores = acc, f1, auc, conf_mat
        scores = acc, conf_mat
        return scores

    def nn_fit(self, x_train, x_val, y_train, y_val):
        # Selects gpu if available else cpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            use_cuda = True
        else:
            self.device = torch.device("cpu")
            use_cuda = False

        # creates model and configuration
        self.model = self.model.to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.model.lr, weight_decay=self.model.wd)

        # Converts data to tensor and creates batches
        x_train = torch.tensor(x_train.astype(np.float32))
        x_val = torch.tensor(x_val.astype(np.float32))
        y_train = torch.tensor(y_train.astype(np.int64))
        y_val = torch.tensor(y_val.astype(np.int64))
        train_ds = TensorDataset(x_train, y_train)
        train_ds_loader = DataLoader(train_ds, batch_size=self.model.batch_size, pin_memory=use_cuda)

        self.train_loss = []
        self.val_loss = []
        for i in range(self.model.n_epochs):
            # training
            for x, y in train_ds_loader:
                # basic configurations before training starts
                optimizer.zero_grad()
                self.model.train()
                x, y = x.to(self.device), y.to(self.device)

                # Running forward part
                y_pred = self.model(x)

                # backward part
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
            # validation
            # basic configurations before validation starts
            self.model.eval()
            with torch.no_grad():
                x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                self.train_loss.append(loss.to('cpu').numpy())
                y_val_pred = self.model(x_val)
                self.val_loss.append(loss_fn(y_val_pred, y_val).to('cpu').numpy())
                print(f'{i}: train_loss={self.train_loss[i]}, val_loss={self.val_loss[i]}')

        self.train_loss = np.array(self.train_loss).flatten()
        self.val_loss = np.array(self.val_loss).flatten()
        self.model = self.model.to('cpu')

    def load_model(self, model_filename):
        if self.model_name.startswith('nn_'):
            self.model.load_state_dict(torch.load(os.path.join(self.models_path, model_filename)))
            self.model = self.model.eval()
        else:
            self.model = load(os.path.join(self.models_path, model_filename))