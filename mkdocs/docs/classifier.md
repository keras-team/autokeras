###_validate
Check `x_train`'s type and the shape of `x_train`, `y_train`.

###read_csv_file
Read the csv file and returns two seperate list containing images name and their labels.

#####Args
* **csv_file_path**: Path to the CSV file.

#####Returns
* **img_file_names**: List containing images names.

* **img_label**: List containing their respective labels.

###read_images
Read the images from the path and return their numpy.ndarray instance. Return a numpy.ndarray instance containing the training data.

#####Args
* **img_file_names**: List containing images names.

* **images_dir_path**: Path to the directory containing images.

###load_image_dataset
Load images from the files and labels from a csv file.
Second, the dataset is a set of images and the labels are in a CSV file. The CSV file should contain two columns whose names are 'File Name' and 'Label'. The file names in the first column should match the file names of the images with extensions, e.g., .jpg, .png. The path to the CSV file should be passed through the `csv_file_path`. The path to the directory containing all the images should be passed through `image_path`.
#####Args
* **csv_file_path**: CSV file path.

* **images_path**: Path where images exist.

#####Returns
* **x**: Four dimensional numpy.ndarray. The channel dimension is the last dimension.

* **y**: The labels.

##ImageClassifier
The image classifier class.
It is used for image classification. It searches convolutional neural network architectures for the best configuration for the dataset.
#####Attributes
* **path**: A path to the directory to save the classifier.

* **y_encoder**: An instance of OneHotEncoder for `y_train` (array of categorical labels).

* **verbose**: A boolean value indicating the verbosity mode.

* **searcher**: An instance of BayesianSearcher. It searches different
    neural architecture to find the best model.

* **searcher_args**: A dictionary containing the parameters for the searcher's __init__ function.

* **augment**: A boolean value indicating whether the data needs augmentation.

###__init__
Initialize the instance.
The classifier will be loaded from the files in 'path' if parameter 'resume' is True. Otherwise it would create a new one.
#####Args
* **verbose**: A boolean of whether the search process will be printed to stdout.

* **path**: A string. The path to a directory, where the intermediate results are saved.

* **resume**: A boolean. If True, the classifier will continue to previous work saved in path.
    Otherwise, the classifier will start a new search.

* **augment**: A boolean value indicating whether the data needs augmentation.

###fit
Find the best neural architecture and train it.
Based on the given dataset, the function will find the best neural architecture for it. The dataset is in numpy.ndarray format. So they training data should be passed through `x_train`, `y_train`.
#####Args
* **x_train**: A numpy.ndarray instance containing the training data.

* **y_train**: A numpy.ndarray instance containing the label of the training data.

* **time_limit**: The time limit for the search in seconds.

###predict
Return predict results for the testing data.

#####Args
* **x_test**: An instance of numpy.ndarray containing the testing data.

#####Returns
###evaluate
Return the accuracy score between predict value and `y_test`.

###final_fit
Final training after found the best architecture.

#####Args
* **x_train**: A numpy.ndarray of training data.

* **y_train**: A numpy.ndarray of training targets.

* **x_test**: A numpy.ndarray of testing data.

* **y_test**: A numpy.ndarray of testing targets.

* **trainer_args**: A dictionary containing the parameters of the ModelTrainer constructure.

* **retrain**: A boolean of whether reinitialize the weights of the model.

###get_best_model_id
Return an integer indicating the id of the best model.

