###ensure_dir
Create directory if it does not exist

###ensure_file_dir
Create path if it does not exist

##ModelTrainer
A class that is used to train the model.
This class can train a model with dataset and will not stop until getting the minimum loss.
#####Attributes
* **model**: The model that will be trained

* **train_data**: Training data wrapped in batches.

* **test_data**: Testing data wrapped in batches.

* **verbose**: Verbosity mode.

###__init__
Init the ModelTrainer with `model`, `x_train`, `y_train`, `x_test`, `y_test`, `verbose`

###train_model
Train the model.

#####Args
* **max_iter_num**: An integer. The maximum number of epochs to train the model.
    The training will stop when this number is reached.

* **max_no_improvement_num**: An integer. The maximum number of epochs when the loss value doesn't decrease.
    The training will stop when this number is reached.

* **batch_size**: An integer. The batch size during the training.

