# Getting Started

## Installation
The installation of Auto-Keras is the same as other python packages. Notably, currently we only support Python 3.6.

#### Latest Stable Version (`pip` installation):
You can run the following `pip` installation command in your terminal to install the latest stable version.

    pip install autokeras

#### Bleeding Edge Version (manual installation):
If you want to install the latest development version. 
You need to download the code from the GitHub repo and run the following commands in the project directory.

    pip install -r requirements.txt
    python setup.py install
    

## Example

We show an example of image classification on the MNIST dataset, which is a famous benchmark image dataset for hand-written digits classification. Auto-Keras support different types of data inputs. 

#### Data with *numpy array * (.npy) format.

If the images and the labels are already formatted into numpy arrays, you can 

    from keras.datasets import mnist
    from autokeras.classifier import ImageClassifier

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

#### What if your data are images files (*e.g.* .jpg, .png, .bmp)?

You can use our `load_image_dataset` function to load the images and there labels as follows.

    from autokeras.classifier import load_image_dataset
    
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
