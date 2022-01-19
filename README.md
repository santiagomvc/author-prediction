# Author Classification
## Exploratory Analysis
1. Open `exploration.ipynb` and review the exploratory analysis
## Train model
1. Clone this repository into your local environment
2. Add the training data in a `train.csv` file using the format from `train_template.csv`
3. Install the libraries in `requirements.txt` into your local python environment
4. Run `train.py`
### Predict with a trained model
1. Train the model following the instructions above
2. Add the inference data in a `test.csv` file using the format from `test_template.csv`
3. Run `infer.py`
### Note
It WILL take some time to TRAIN and PREDICT with the model due to the enconding creation process. If there're memory errors reduce `batch_size` parameter in `DataLoader` function