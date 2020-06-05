"""shell
pip install autokeras
"""

"""
## A Simple Example with Auto MPG Data Set

Download [Auto MPG Data Set](https://archive.ics.uci.edu/ml/datasets/auto+mpg):
"""

import tensorflow as tf
import pandas as pd
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
dataset_path = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset = dataset.dropna()
dataset.tail()

"""
Make all but the last feature ('Origin') numerical.
"""

column_names.remove('MPG')
data_cols =column_names 
data_type = (len(data_cols)-1) * ['numerical'] + ['categorical']
data_type = dict(zip(data_cols, data_type))

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_dataset.describe()

import autokeras as ak

regressor = ak.StructuredDataRegressor(max_trials=100, column_names=data_cols, column_types=data_type)
regressor.fit(x=train_dataset.drop(columns=['MPG']), y=train_dataset['MPG'])
# Evaluate the accuracy of the found model.
print('Accuracy: {accuracy}'.format(
    accuracy=regressor.evaluate(x=test_dataset.drop(columns=['MPG']), y=test_dataset['MPG'])))

"""
Accuracy: [9.906872749328613, 9.715665]
"""

model = regressor.export_model()
tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True)

"""
![Network Topology found by autokeras](Reg_Network.png)
"""

