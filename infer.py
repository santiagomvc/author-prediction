# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, dataloader
from transformers import DistilBertTokenizer, DistilBertModel
from utils.model_utils import Modeler, LstmNet
from utils.model_utils import split_data

# Read training data
df = pd.read_csv('data/test.csv', index_col=0)
labels = ['defoe', 'dickens', 'doyle', 'twain']

# Preprocess text
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenized_texts = tokenizer(df['text'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=90)
texts_ds = TensorDataset(tokenized_texts.input_ids)
texts_ds_loader = DataLoader(texts_ds, batch_size=100)

# loading and applying pretrained model
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Encoding texts using pretrained language models
encodings = []
i = 0
for text in texts_ds_loader:
    print(f'Encoding {i} out of 9')
    i += 1
    x = model(text[0]).last_hidden_state.detach().numpy()
    encodings.append(x)
np_encs = np.array(encodings)
x = np_encs.reshape((-1,np_encs.shape[2],np_encs.shape[3]))
x = x[:,:50,:]

# Build model
# Running models
params_list = [
    {
        'n_epochs': [20],
        'batch_size': [512],
        'hid_dim': [64],
        'num_layers': [1],
        'lr': [0.0005],
        'wd': [0.002],
        'pos_weight': [1],
        'dropout': [0.95],
        'state_comb': ['all'],
        'bidirectional': [True],
        'n_steps': [x.shape[1]],
        'feat_dim': [x.shape[2]],
        'y_dim': [4]
    }
]
params_grid_list = [
    list(ParameterGrid(params_dict)) for params_dict in params_list
]

lstm_filename = 'nn_lstm.pt' # rename downloaded file to this
lstm_modeler = Modeler('nn_lstm', LstmNet, params_grid_list[0][0], '')
lstm_modeler.load_model(lstm_filename)

# predicts over the data
df['preds'] = lstm_modeler.predict(x)
df['author'] = df['preds'].apply(lambda x: labels[x])
df = df.drop(['preds', 'text'], axis=1)
# saves data
df.to_csv('results.csv')