# Getting Started

---

## Installation
The installation of Auto-Keras is the same as other python packages. 

**Note:** currently, Auto-Keras is only compatible with: **Python 3.6**.

### Latest Stable Version (`pip` installation):
You can run the following `pip` installation command in your terminal to install the latest stable version.

    pip install autokeras

### Bleeding Edge Version (manual installation):
If you want to install the latest development version. 
You need to download the code from the GitHub repo and run the following commands in the project directory.

    pip install -r requirements.txt
    python setup.py install




## A Simple Example

We show an example of image classification on the MNIST dataset, which is a famous benchmark image dataset for hand-written digits classification. Auto-Keras supports different types of data inputs. 


### Data with numpy array (.npy) format. [[source]](https://github.com/jhfjhfj1/autokeras/blob/master/examples/a_simple_example/mnist.py)

If the images and the labels are already formatted into numpy arrays, you can 

    from keras.datasets import mnist
    from autokeras.image.image_supervised import ImageClassifier

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


### What if your data are raw image files (*e.g.* .jpg, .png, .bmp)? [[source]](https://github.com/jhfjhfj1/autokeras/blob/master/examples/a_simple_example/load_raw_image.py)

You can use our `load_image_dataset` function to load the images and their labels as follows.

    from autokeras.image.image_supervised import load_image_dataset
    
    x_train, y_train = load_image_dataset(csv_file_path="train/label.csv",
                                          images_path="train")
    print(x_train.shape)
    print(y_train.shape)
    
    x_test, y_test = load_image_dataset(csv_file_path="test/label.csv",
                                        images_path="test")
    print(x_test.shape)
    print(y_test.shape)

The argument `csv_file_path` is the path to the CSV file containing the image file names and their corresponding labels. Both csv files and the raw image datasets could be downloaded from [link](https://drive.google.com/a/tamu.edu/file/d/10TyvztrdL0fBaFlgGaqoRTlS3ts4faM8/view?usp=sharing).
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


This CSV file for train or test can be created from folders containing images of a specific class (meaning label):
```
train
└───class_1
│   │   class_1_image_1.png
│   │   class_1_image_2.png
|   |   ...
└───class_2
    │   class_2_image_1.png
    │   class_2_image_2.png
    |   ...
```
The code below shows an example of how to create the CSV:
```
train_dir = 'train' # Path to the train directory
class_dirs = [i for i in os.listdir(path=train_dir) if os.path.isdir(os.path.join(train_dir, i))]
 with open('train/label.csv', 'w') as train_csv:
    fieldnames = ['File Name', 'Label']
    writer = csv.DictWriter(train_csv, fieldnames=fieldnames)
    writer.writeheader()
    label = 0
    for current_class in class_dirs:
        for image in os.listdir(os.path.join(train_dir, current_class)):
            writer.writerow({'File Name': str(image), 'Label':label})
        label += 1
    train_csv.close()
```



## Portable Models

### How to export keras models?
    clf.load_searcher().load_best_model().produce_keras_model().save('my_model.h5')
This uses the keras function model.save() to export a single HDF5 file containing the architecture of the model, the weights of the model, the training configuration, and the state of the optimizer. See https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model

**Note:** This is being built into AutoKeras as ImageClassifier().export_keras_model() 


### How to export Portable model? [[source]](https://github.com/jhfjhfj1/autokeras/blob/master/examples/portable_models/portable_load.py)

    from autokeras import ImageClassifier
    clf = ImageClassifier(verbose=True, augment=False)
    clf.export_autokeras_model(model_file_name)
The model will be stored into the path `model_file_name`. 


### How to load exported Portable model? [[source]](https://github.com/jhfjhfj1/autokeras/blob/master/examples/portable_models/portable_load.py)

    from autokeras.utils import pickle_from_file
    model = pickle_from_file(model_file_name)
    results = model.evaluate(x_test, y_test)
    print(results)
    
The model will be loaded from the path `model_file_name` and then you can use the functions listed in `PortableImageSupervised`.

	



## Model Visualizations


### How to visualize keras models? 

This is not specific to AutoKeras, however, the following will generate a .PNG visualization of the best model found by AutoKeras:

    from keras.models import load_model
    model = load_model('my_model.h5') #See 'How to export keras models?' to generate this file before loading it.
    from keras.utils import plot_model
    plot_model(model, to_file='my_model.png')

    

### How to visualize the best selected architecture? [[source]](https://github.com/jhfjhfj1/autokeras/blob/master/examples/visualizations/visualize.py)


While trying to create a model, let's say an Image classifier on MNIST, there is a facility for the user to visualize a .PDF depiction of the best architecture that was chosen by autokeras, after model training is complete. 

Prerequisites : 
1) graphviz must be installed in your system. Refer [Installation Guide](https://graphviz.gitlab.io/download/)  
2) Additionally, also install "graphviz" python package using pip / conda


    pip:  pip install graphviz
    
    conda : conda install -c conda-forge python-graphviz

If the above installations are complete, proceed with the following steps :

Step 1 : Specify a *path* before starting your model training

    clf = ImageClassifier(path="~/automodels/",verbose=True, augment=False) # Give a custom path of your choice
    clf.fit(x_train, y_train, time_limit=30 * 60)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)

Step 2 : After the model training is complete, run *examples/visualize.py*, whilst passing the same *path* as parameter

    if __name__ == '__main__':
        visualize('~/automodels/')




## Net Modules


### MlpModule tutorial. [[source]](https://github.com/jhfjhfj1/autokeras/blob/master/examples/net_modules/mlp_module.py)


`MlpGenerator` in `net_module.py` is a child class of `Networkmodule`. It can generates neural architecture with MLP modules 


Normally, there's two place to call the MlpGenerator, one is call `MlpGenerator.fit` while the other is `MlpGenerator.final_fit`.

For example, in a image classification class `ImageClassifier`, one can initialize the cnn module as:

```python
mlpModule = MlpModule(loss, metric, searcher_args, path, verbose)
```
Where:
* `loss` and `metric` determines by the type of training model(classification or regression or others)
* `search_args` can be referred in `search.py`
* `path` is the path to store the whole searching process and generated model.
* `verbose` is a boolean. Setting it to true prints to stdout.

Then, for the searching part, one can call:
```python
mlpModule.fit(n_output_node, input_shape, train_data, test_data, time_limit=24 * 60 * 60)
```
where:
* n_output_node: A integer value represent the number of output node in the final layer.
* input_shape: A tuple to express the shape of every train entry. For example,
                MNIST dataset would be (28,28,1).
* train_data: A PyTorch DataLoader instance representing the training data.
* test_data: A PyTorch DataLoader instance representing the testing data.
* time_limit: A integer value represents the time limit on searching for models.

And for final testing(testing the best searched model), one can call:
```python
mlpModule.final_fit(train_data, test_data, trainer_args=None, retrain=False)
```
where:
* train_data: A DataLoader instance representing the training data.
* test_data: A DataLoader instance representing the testing data.
* trainer_args: A dictionary containing the parameters of the ModelTrainer constructor.
* retrain: A boolean of whether reinitialize the weights of the model.





### CnnModule tutorial. [[source]](https://github.com/jhfjhfj1/autokeras/blob/master/examples/net_modules/cnn_module.py)


`CnnGenerator` in `net_module.py` is a child class of `Networkmodule`. It can generates neural architecture with basic cnn modules
and the ResNet module. 


Normally, there's two place to call the CnnGenerator, one is call `CnnGenerator.fit` while the other is `CnnGenerator.final_fit`.

For example, in a image classification class `ImageClassifier`, one can initialize the cnn module as:

```python
from autokeras import CnnModule
from autokeras.nn.loss_function import classification_loss
from autokeras.nn.metric import Accuracy

TEST_FOLDER = "test"
cnnModule = CnnModule(loss=classification_loss, metric=Accuracy, searcher_args={}, path=TEST_FOLDER, verbose=False)
```
Where:
* `loss` and `metric` determines by the type of training model(classification or regression or others)
* `search_args` can be referred in `search.py`
* `path` is the path to store the whole searching process and generated model.
* `verbose` is a boolean. Setting it to true prints to stdout.

Then, for the searching part, one can call:
```python
cnnModule.fit(n_output_node, input_shape, train_data, test_data, time_limit=24 * 60 * 60)
```
where:
* n_output_node: A integer value represent the number of output node in the final layer.
* input_shape: A tuple to express the shape of every train entry. For example,
                MNIST dataset would be (28,28,1).
* train_data: A PyTorch DataLoader instance representing the training data.
* test_data: A PyTorch DataLoader instance representing the testing data.
* time_limit: A integer value represents the time limit on searching for models.

And for final testing(testing the best searched model), one can call:
```python
cnnModule.final_fit(train_data, test_data, trainer_args=None, retrain=False)
```
where:
* train_data: A DataLoader instance representing the training data.
* test_data: A DataLoader instance representing the testing data.
* trainer_args: A dictionary containing the parameters of the ModelTrainer constructor.
* retrain: A boolean of whether reinitialize the weights of the model.



## Task Modules
 

### Automated text classifier tutorial. [[source]](https://github.com/jhfjhfj1/autokeras/blob/master/examples/task_modules/text/text.py)

Class `TextClassifier` and `TextRegressor` are designed for automated generate best performance cnn neural architecture
for a given text dataset. 


```python
    clf = TextClassifier(verbose=True)
    clf.fit(x=x_train, y=y_train, batch_size=10, time_limit=12 * 60 * 60)
```

* x_train: string format text data
* y_train: int format text label

After searching the best model, one can call `clf.final_fit` to test the best model found in searching.


**Notes:** Preprocessing of the text data:
* Class `TextClassifier` and `TextRegressor` contains a pre-process of the text data. Which means the input data
should be in string format. 
* The default pre-process model uses the [glove6B model](https://nlp.stanford.edu/projects/glove/) from Stanford NLP. 
* To change the default setting of the pre-process model, one need to change the corresponding variable:
`EMBEDDING_DIM`, `PRE_TRAIN_FILE_LINK`, `PRE_TRAIN_FILE_LINK`, `PRE_TRAIN_FILE_NAME` in `constant.py`.

 
 

### Automated tabular classifier tutorial. [[source]](https://github.com/jhfjhfj1/autokeras/tree/master/examples/task_modules/tabular)


Class `TabularClassifier` and `TabularRegressor` are designed for automated generate best performance shallow/deep architecture
for a given tabular dataset. (Currently, theis module only supports lightgbm classifier and regressor.)


```python
    clf = TabularClassifier(verbose=True)
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60, data_info=datainfo)
```

* x_train: string format text data
* y_train: int format text label
* data_info: a numpy.array describing the feature types (time, numerical or categorical) of each column in x_train.


**Notes:** Preprocessing of the tabular data:
* Class `[TabularPreprocessor]` involves several automated feature preprocessing and engineering operation for tabular data . 
*The input data should be in numpy array format for the class `TabularClassifier` and `TabularRegressor` .
 
 
 
## Pretrained Models
 

### Object detection tutorial. [[source]](https://github.com/jhfjhfj1/autokeras/blob/master/examples/pretrained_models/object_detection/object_detection_example.py)

#### by Wuyang Chen from [Dr. Atlas Wang's group](http://www.atlaswang.com/) at CSE Department, Texas A&M.

`ObjectDetector` in `object_detector.py` is a child class of `Pretrained`. Currently it can load a pretrained SSD model ([Liu, Wei, et al. "Ssd: Single shot multibox detector." European conference on computer vision. Springer, Cham, 2016.](https://arxiv.org/abs/1512.02325)) and find object(s) in a given image.

Let's first import the ObjectDetector and create a detection model (```detector```) with
```python
from autokeras.pretrained.object_detector import ObjectDetector
detector = ObjectDetector()
```
**Note:**  the ```ObjectDetector``` class can automatically detect the existance of available cuda device(s), and use the device if exists.

Second, you will want to load the pretrained weights for your model:
```python
detector.load()
```
This line will automatically download and load the weights into ```detector```.
Finally you can make predictions against an image:
```python
    results = detector.predict("/path/to/images/000001.jpg", output_file_path="/path/to/images/")
```
Function ```detector.predict()``` requires the path to the image. If the ```output_file_path``` is not given, the ```detector``` will just return the numerical results as a list of dictionaries. Each dictionary is like {"left": int, "top": int, "width": int, "height": int: "category": str, "confidence": float}, where ```left``` and ```top``` is the (left, top) coordinates of the bounding box of the object and ```width``` and ```height``` are width and height of the box. ```category``` is a string representing the class the object belongs to, and the confidence can be regarded as the probability that the model believes its prediction is correct. If the ```output_file_path``` is given, then the results mentioned above will be plotted and saved in a new image file with suffix "_prediction" into the given ```output_file_path```. If you run the example/object_detection/object_detection_example.py, you will get result
```[{'category': 'person', 'width': 331, 'height': 500, 'left': 17, 'confidence': 0.9741123914718628, 'top': 0}]```

 
 
 
 
 
 
 
 
 
 

<!-- [Data with numpy array (.npy) format.]: https://github.com/jhfjhfj1/autokeras/blob/master/examples/a_simple_example/mnist.py
[What if your data are raw image files (*e.g.* .jpg, .png, .bmp)?]: https://github.com/jhfjhfj1/autokeras/blob/master/examples/a_simple_example/load_raw_image.py
[How to export Portable model]: https://github.com/jhfjhfj1/autokeras/blob/master/examples/portable_models/portable_load.py
[How to load exported Portable model?]: https://github.com/jhfjhfj1/autokeras/blob/master/examples/portable_models/portable_load.py
[How to visualize the best selected architecture?]: https://github.com/jhfjhfj1/autokeras/blob/master/examples/visualizations/visualize.py
[MlpModule tutorial]: https://github.com/jhfjhfj1/autokeras/blob/master/examples/net_modules/mlp_module.py
[CnnModule tutorial]: https://github.com/jhfjhfj1/autokeras/blob/master/examples/net_modules/cnn_module.py
[Automated text classifier tutorial]: https://github.com/jhfjhfj1/autokeras/blob/master/examples/task_modules/text/text.py
[Automated tabular classifier tutorial]: https://github.com/jhfjhfj1/autokeras/tree/master/examples/task_modules/tabular
[Object Detection tutorial]: https://github.com/jhfjhfj1/autokeras/blob/master/examples/pretrained_models/object_detection/object_detection_example.py -->
[TabularPreprocessor]: https://github.com/jhfjhfj1/autokeras/blob/master/autokeras/tabular/tabular_preprocessor.py

