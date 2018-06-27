##NoImprovementError
###__init__
##EarlyStop
###__init__
###on_train_begin
###on_epoch_end
##ModelTrainer
A class that is used to train model
This class can train a model with dataset and will not stop until getting minimum loss
####Attributes
**model**: the model that will be trained

**x_train**: the input train data

**y_train**: the input train data labels

**x_test**: the input test data

**y_test**: the input test data labels

**verbose**: verbosity mode

###__init__
Init ModelTrainer with model, x_train, y_train, x_test, y_test, verbose

###train_model
