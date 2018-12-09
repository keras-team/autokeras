# Getting Started

## Installation
The installation of Auto-Keras is the same as other python packages. 

**Note:** currently, Auto-Keras is only compatible with: **Python 3.6**.

#### Latest Stable Version (`pip` installation):
You can run the following `pip` installation command in your terminal to install the latest stable version.

    pip install autokeras

#### Bleeding Edge Version (manual installation):
If you want to install the latest development version. 
You need to download the code from the GitHub repo and run the following commands in the project directory.

    pip install -r requirements.txt
    python setup.py install
    

## Example

We show an example of image classification on the MNIST dataset, which is a famous benchmark image dataset for hand-written digits classification. Auto-Keras supports different types of data inputs. 

#### Data with numpy array (.npy) format.

If the images and the labels are already formatted into numpy arrays, you can 

    from keras.datasets import mnist
    from autokeras.image_supervised import ImageClassifier

    if __name__ == '__main__':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.reshape(x_test.shape + (1,))

        clf = ImageClassifier(verbose=True)
        clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
        clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
        y = clf.evaluate(x_test, y_test)
        print(y)
        
In the example above, the images and the labels are already formatted into numpy arrays.

#### What if your data are raw image files (*e.g.* .jpg, .png, .bmp)?

You can use our `load_image_dataset` function to load the images and their labels as follows.

    from autokeras.image_supervised import load_image_dataset
    
    x_train, y_train = load_image_dataset(csv_file_path="train/label.csv",
                                          images_path="train")
    print(x_train.shape)
    print(y_train.shape)
    
    x_test, y_test = load_image_dataset(csv_file_path="test/label.csv",
                                        images_path="test")
    print(x_test.shape)
    print(y_test.shape)

The argument `csv_file_path` is the path to the CSV file containing the image file names and their corresponding labels.
Here is an example of the csv file.


    File Name,Label
    00000.jpg,5
    00001.jpg,0
    00002.jpg,4
    00003.jpg,1
    00004.jpg,9
    00005.jpg,2
    00006.jpg,1
    ...


The second argument `images_path` is the path to the directory containing all the images with those file names listed in the CSV file.
The returned values `x_train` and `y_train` are the numpy arrays,
which can be directly feed into the `fit` function of `ImageClassifier`.

#### How to export keras models?

    from autokeras import ImageClassifier
    clf = ImageClassifier(verbose=True, augment=False)
    clf.load_searcher().load_best_model().produce_keras_model().save('my_model.h5')

This uses the keras function model.save() to export a single HDF5 file containing the architecture of the model, the weights of the model, the training configuration, and the state of the optimizer. See https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
Note: This is being built into AutoKeras as ImageClassifier().export_keras_model() 
	
#### How to visualize keras models?

This is not specific to AutoKeras, however, the following will generate a .PNG visualization of the best model found by AutoKeras:

    from keras.models import load_model
    model = load_model('my_model.h5') #See 'How to export keras models?' to generate this file before loading it.
    from keras.utils import plot_model
    plot_model(model, to_file='my_model.png')
    
#### How to visualize the best selected architecture ?

While trying to create a model, let's say an Image classifier on MNIST, there is a facility for the user to visualize a .PDF depiction of the best architecture that was chosen by autokeras, after model training is complete. 

Prerequisites : 
1) graphviz must be installed in your system. Refer [Installation Guide](https://graphviz.gitlab.io/download/)  
2) Additionally, also install "graphviz" python package using pip / conda

pip : pip install graphviz

conda : conda install -c conda-forge python-graphviz

If the above installations are complete, proceed with the following steps :

Step 1 : Specify a *path* before starting your model training

    clf = ImageClassifier(path="~/automodels/",verbose=True, augment=False) # Give a custom path of your choice
    clf.fit(x_train, y_train, time_limit=30 * 60)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)

Step 2 : After the model training is complete, run *examples/visualize.py*, whilst passing the same *path* as parameter

    if __name__ == '__main__':
        visualize('~/automodels/')


