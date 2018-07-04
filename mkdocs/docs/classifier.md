###_validate
Check x_train's type and the shape of x_train, y_train.

###read_csv_file
Read the cvs file and returns two seperate list containing images name and their labels

####Args
**csv_file_path**: Path to the CVS file.

####Returns
img_file_names list containing images names and img_label list containing their respective labels.
###read_images
Reads the images from the path and return there numpy.ndarray instance

####Args
**img_file_names**: List containing images names

**images_dir_path**: Path to directory containing images

###load_image_dataset
Load images from the files and labels from a csv file.
Second, the dataset is a set of images and the labels are in a CSV file. The CSV file should contain two columns whose names are 'File Name' and 'Label'. The file names in the first column should match the file names of the images with extensions, e.g., .jpg, .png. The path to the CSV file should be passed through the csv_file_path. The path to the directory containing all the images should be passed through image_path.
####Args
**csv_file_path**: CVS file path.

**images_path**: Path where images exist.

####Returns
x: Four dimensional numpy.ndarray. The channel dimension is the last dimension. y: The labels.
##ImageClassifier
The image classifier class.
It is used for image classification. It searches convolutional neural network architectures for the best configuration for the dataset.
####Attributes
**path**: A path to the directory to save the classifier.

**y_encoder**: An instance of OneHotEncoder for y_train (array of categorical labels).

**verbose**: A boolean value indicating the verbosity mode.

**searcher**: An instance of BayesianSearcher. It search different
    neural architecture to find the best model.

**searcher_args**: A dictionary containing the parameters for the searcher's __init__ function.

###__init__
Initialize the instance.
The classifier will be loaded from the files in 'path' if parameter 'resume' is True. Otherwise it would create a new one.
####Args
**verbose**: An boolean of whether the search process will be printed to stdout.

**path**: A string. The path to a directory, where the intermediate results are saved.

**resume**: An boolean. If True, the classifier will continue to previous work saved in path.
    Otherwise, the classifier will start a new search.

###fit
Find the best neural architecture and train it.
Based on the given dataset, the function will find the best neural architecture for it. The dataset is in numpy.ndarray format. So they training data should be passed through x_train, y_train.
####Args
**x_train**: An numpy.ndarray instance contains the training data.

**y_train**: An numpy.ndarray instance contains the label of the training data.

**time_limit**: The time limit for the search in seconds.

###predict
Return predict result for the testing data.

####Args
**x_test**: An instance of numpy.ndarray contains the testing data.

####Returns
An numpy.ndarray containing the results.
###summary
Print the summary of the best model.

###evaluate
Return the accuracy score between predict value and test_y.

###final_fit
Final training after found the best architecture.

####Args
**x_train**: An numpy.ndarray of training data.

**y_train**: An numpy.ndarray of training targets.

**x_test**: An numpy.ndarray of testing data.

**y_test**: An numpy.ndarray of testing targets.

**trainer_args**: A dictionary containing the parameters of the ModelTrainer constructure.

**retrain**: A boolean of whether reinitialize the weights of the model.

###export_keras_model
Export the searched model as a Keras saved model.

####Args
**path**: A string. The path to the file to save.

**model_id**: A integer. If not provided, the function will export the best model.

###get_best_model_id
Returns: An integer. The best model id.

