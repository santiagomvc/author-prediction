# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader, dataloader
from transformers import DistilBertTokenizer, DistilBertModel
from utils.model_utils import Modeler, LstmNet
from utils.model_utils import split_data

# Read training data
df = pd.read_csv('data/train.csv', index_col=0)
labels = ['defoe', 'dickens', 'doyle', 'twain']
df['author'] = df['author'].apply(lambda x: labels.index(x))

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
    print(f'Encoding {i} out of 39')
    i += 1
    x = model(text[0]).last_hidden_state.detach().numpy()
    encodings.append(x)
np_encs = np.array(encodings)
x = np_encs.reshape((-1,np_encs.shape[2],np_encs.shape[3]))
# np.save('distilbert_enc_90.npy', x)
# x = np.load('distilbert_enc_90.npy')
x = x[:,:50,:]

# split data
_, y, train_idx, val_idx = split_data(df, 'author', .05)
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# Build model
# Running models
algos = {
    'nn_lstm': LstmNet
}
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
        'n_steps': [x_train.shape[1]],
        'feat_dim': [x_train.shape[2]],
        'y_dim': [4]
    }
]
params_grid_list = [
    list(ParameterGrid(params_dict)) for params_dict in params_list
]

for algo_name, algo_class, params_grid in zip(algos.keys(), algos.values(), params_grid_list):
    for params in params_grid:
        print(algo_name, params)
        modeler = Modeler(algo_name, algo_class, params, '')
        modeler.train(x_train, x_val, y_train, y_val) 
        val_metrics = modeler.score(x_val, y_val)
        y_val_preds_probas = modeler.predict_proba(x_val) 
        if algo_name.startswith('nn_'):
            plt.show()
            plt.plot(list(range(len(modeler.train_loss))), modeler.train_loss, label='train_loss')
            plt.plot(list(range(len(modeler.val_loss))), modeler.val_loss, label='val_loss')
            plt.legend()
            plt.show()
            plt.clf()
        print(f'Valid Acc: {val_metrics[0]}')
        val_cm_plot = ConfusionMatrixDisplay(val_metrics[1], display_labels=(0,1,2,3))
        val_cm_plot.plot()
        plt.show()
        plt.clf()
