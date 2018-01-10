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

**training_losses**: a list to store all losses during training

**minimum_loss**: the minimum loss during training

**_no_improvement_count**: the number of iterations that don't improve the result

###__init__
Init ModelTrainer with model, x_train, y_train, x_test, y_test, verbose

###_converged
Return whether the training is converged

###train_model
Train the model with dataset and return the minimum_loss

