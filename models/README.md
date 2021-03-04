# Models

This folder contains the XGBoost trained models with hopefully relevant names.

## Use the models for prediction in Python

The trained model can be used to evaluate their performances or to use them for prediction (determine the position of seizure onset).

```python
# Import XGBoost
import xgboost as xgb

# Import NumPy
import numpy as np

# Set model's file path
model_file = '/home_nfs/stragierv/TUH_SZ_v1.5.2/'
             'Epilepsy_Seizure_Detection/models/'
             'all_chanels_initials.model'

# Set the features path
features_file = '/home_nfs/stragierv/TUH_SZ_v1.5.2/'
                'Epilepsy_Seizure_Detection/one_hot_vectors/'
                'X_train.npy'

# Load the model in a Booster
bst = xgb.Booster(model_file=model_file)

# Load the features (one hot vectors) to classify
X_train = np.load(features_file)

# Soft predict the class (0: background, 1: seizure)
y_pred = bst.predict(xgb.DMatrix(X_train))

# Dewindow the prediction
# Assemption: step = 1 s and window = 4 s
n_pad = 3

# Resize the prediction vector to match
# the original duration:
new_shape = list(y_pred.shape)
new_shape[-1] = new_shape[-1] + n_pad
# The new end of the vector is filled with zeros
y_pred.resize(tuple(new_shape))

# Roll and add to the original prediction vector
# Example
# 1 2 3 0 0
# 1 3 5 3 0
# 1 3 6 5 3
y_dewin = np.copy(y_pred)
for index in range(n_pad):
    y_dewin += np.roll(y_pred, index + 1)

# Correct probabilities on the edges (pads)
for index in range(int(max(0, (n_pad - 1)))):
    y_dewin[index + 1] /= (index + 2)
    y_dewin[- (index + 2)] /= (index + 2)

# Correct probabilities in the body
y_dewin[n_pad:-n_pad] /= (n_pad + 1)

# y_dewin now reflects the probability of each second
# of being a seizure event or a background event.
# A threshold can be applied as follow:
y_pred = np.where(y_dewin > 0.5, True, False)
```
