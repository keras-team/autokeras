from abc import abstractmethod

import numpy as np
from autokeras.image_supervised import ImageSupervised, ImageClassifier, PortableClass
from autokeras.preprocessor import OneHotEncoder, DataTransformer
from autokeras.metric import Accuracy, MSE
from autokeras.loss_function import classification_loss, regression_loss

#Contributed by Werner G. Krebs, Ph.D. of Acculation Inc in the hopes that it will be useful and 
#inspire others to contribute a better 1D auto-classification module.

#(C)2018 Acculation Inc

#This is FREE SOFTWARE Contributed under MIT's software license. AS SUCH IT PROVIDED WITHOUT WARRANTY
#OF ANY KIND. See the MIT LICENSE file for complete legal details.

#Quick and dirty "Generalized" classifier for 1D information.
#This can classify most tabular information as long as all information is numeric (and classes are integer).

#There already has an Image1D request (intended for non-image data) which this was closest to.
#To get to the Genealized Classifier, non-numeric data (e.g., non-numeric labels) would need to be
#reassigned numeric labels temporarily, and then run through this.

#Autokeras guidelines say that new classifiers should inherit supervised, but it actually makes more
#sense to inherit ImageSupervised.

#It turns out ImageSupervised (a 2D classifier) can do an automatic search of network topology
#provided each dimension is at least 8 (so 8x8 images, or 64 minimum).

#Consequently, one can simply take 1Dimensional numeric data, zero pad it out to 64 values if it is less than
#64, pad it up to nearest multiple of 8 if it is not divisible by 8, and then reshape as a 2D matrix
#of size 8 by something.

#Image2DClassifier can then be called, usually successfully.

#NOW, DOES THIS WORK WELL?

#Tests against other auto-learn systems like auto-wiki suggests it performs about equally badly on
#data with a real danger of overfitting. (In these tricky cases, a manual, classical 1D dense
#tensorflow topology works better. Even better, hybrid methods where an expert in the data set has
#manually eliminated (non-neural) models that are fitting noise, combined with classic 1D Tensorflow
#topologies massively outperform this system.

#Classic tensorflow 1D topology outperforms this on the simple pima indians example, with the network
#topology widely found in examples on the web. However, in those examples someone presumable spent
#at least a little time experimenting with different size settings for the 3 layers.

#However, with this class, autokeras DOES work, it does find network topologies that do a fair job
#classifying, and it does find network parameters automatically (even if those network parameters
#are more suited to 2D rather than 1D data.)

#THAT BEING SAID, THIS IS INTENDED A BASELINE THAT IT IS HOPED OTHERS WILL IMPROVE UPON.
#THERE ARE LOW-HANGING FRUITS EVEN LARGELY KEEPING THE EXISTING 2D CODE THAT COULD
#GREATLY IMPROVE RESULTS WITH MINIMAL EFFORT.

#Some low-hanging fruits to consider for futher improving this module:
#Hints to Bayesian network topology generator in autokeras that it is, in reality, optimizing 1D
#rather than 2D, and that it should generate trial network topologies better optimized for 1D data,
#such as those that heavily use dense layers rather than 2Dconvolution, etc.

#Additional possible improvement for handling 1D Time Series data:
#It is assumed the 2D shape will not be important for 1D classification problems.
#(For time series data, this may not be quite true, as when reshaping of time-series data from 1D to 2D,
#the 2D shape dimension might correspond to the lag parameter in the traditional time-series modules.
#The dimension of 8 here would suggest that data points 8 time periods apart are somehow related
#(as with a lag). However, other lags may be more optimal with other data. Future versions might
#allow this parameter to be adjusted for working with time-series data where a lag parameter might be important.

def _validate(x_train, y_train):
    """Check `x_train`'s type and the shape of `x_train`, `y_train`."""
    try:
        x_train = x_train.astype('float64')
    except ValueError:
        raise ValueError('x_train should only contain numerical data.')

    if len(x_train.shape) != 2:
        raise ValueError('x_train should have exactly 2 dimensions.')

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError('x_train and y_train should have the same number of instances.')



class Image1DSupervised(ImageSupervised):
    """1D data classifier. Hack to convert 1D data to faux images, so that autokeras can build automatic networks
       on any kind of numeric data using the existing 2D functionality.

       A generalized classifier could now be built from this if it temporarily reassigned any text strings
       to numeric lables and than ran through this.

    Attributes:
        path: A path to the directory to save the classifier.
        y_encoder: An instance of OneHotEncoder for `y_train` (array of categorical labels).
        verbose: A boolean value indicating the verbosity mode.
        searcher: An instance of BayesianSearcher. It searches different
            neural architecture to find the best model.
        searcher_args: A dictionary containing the parameters for the searcher's __init__ function.
        augment: A boolean value indicating whether the data needs augmentation.  If not define, then it
                will use the value of Constant.DATA_AUGMENTATION which is True by default.
    """

    def __init__(self, verbose=False, path=None, resume=False, searcher_args=None, augment=None):
        """Initialize the instance.

        The classifier will be loaded from the files in 'path' if parameter 'resume' is True.
        Otherwise it would create a new one.

        Args:
            verbose: A boolean of whether the search process will be printed to stdout.
            path: A string. The path to a directory, where the intermediate results are saved.
            resume: A boolean. If True, the classifier will continue to previous work saved in path.
                Otherwise, the classifier will start a new search.
            augment: A boolean value indicating whether the data needs augmentation. If not define, then it
                will use the value of Constant.DATA_AUGMENTATION which is True by default.

        """
        super().__init__(verbose)

    def reshapeTo2D(self, X):
        """Reshapes 1D data to be compatible with 2D image classifier.
        Normally this is an internal function, unless using saved keras models, in which case
        you ight wish to call it.

        Args:
            A 2-Dimensional numpy training array

        Returns:
            A 4-dimension numpy training array suitable for use in 2D Classifier functions
        """

        #so data is already two-dimensional in that the x-dimension are training rows
        #and y-dimension are the 1D dimensional input data to the classifier.
        #In reality we are reshaping from 2D to 3D.

        len = X.shape[1]
        padToLen = len
        if padToLen<64: #needs to be at least 8x8 for 2D classification to work
            padToLen = 64
        elif padToLen % 8 > 0:
            padToLen += 8 - padToLen % 8 #make padToLen be divisble by 8.

        Xout = None
        if padToLen != len:
            X1 = np.empty((X.shape[0],padToLen))
            X1[:,0:len] = X
            Xout=X1
        else:
            Xout = copy.copy(X)
            
        Xout = Xout.reshape(Xout.shape[0],8,padToLen//8) #reshape to 2D image with all dimensions at least 8.
        Xout = Xout.reshape(Xout.shape + (1,))
        return Xout


    def fit(self, x_train=None, y_train=None, time_limit=None):
        """Find the best neural architecture and train it.

        Based on the given dataset, the function will find the best neural architecture for it.
        The dataset is in numpy.ndarray format.
        So they training data should be passed through `x_train`, `y_train`.

        Args:
            x_train: A numpy.ndarray instance containing the training data.
            y_train: A numpy.ndarray instance containing the label of the training data.
            time_limit: The time limit for the search in seconds.
        """
        if y_train is None:
            y_train = []
        if x_train is None:
            x_train = []

        x_train = np.array(x_train)
        y_train = np.array(y_train).flatten()

        _validate(x_train, y_train)
        super().fit(x_train=self.reshapeTo2D(np.array(x_train)),y_train=y_train,time_limit=time_limit)

    def predict(self, x_test):
        """Return predict results for the testing data.

        Args:
            x_test: An instance of numpy.ndarray containing the testing data.

        Returns:
            A numpy.ndarray containing the results.
        """
        return super().predict(x_test=self.reshapeTo2D(np.array(x_test)))

    def evaluate(self, x_test, y_test):
        """Return the accuracy score between predict value and `y_test`."""
        y_predict = self.predict(x_test)
        return self.metric().evaluate(y_test, y_predict)

    def final_fit(self, x_train, y_train, x_test, y_test, trainer_args=None, retrain=False):
        """Final training after found the best architecture.

        Args:
            x_train: A numpy.ndarray of training data.
            y_train: A numpy.ndarray of training targets.
            x_test: A numpy.ndarray of testing data.
            y_test: A numpy.ndarray of testing targets.
            trainer_args: A dictionary containing the parameters of the ModelTrainer constructor.
            retrain: A boolean of whether reinitialize the weights of the model.
        """
        super().final_fit(self.reshapeTo2D(np.array(x_train)),y_train=y_train,x_test=self.reshapeTo2D(np.array(x_test)),y_test=y_test,trainer_args=trainer_args,retrain=retrain)

class Image1DClassifier(Image1DSupervised):
    @property
    def loss(self):
        return classification_loss

    def transform_y(self, y_train):
        # Transform y_train.
        if self.y_encoder is None:
            self.y_encoder = OneHotEncoder()
            self.y_encoder.fit(y_train)
        y_train = self.y_encoder.transform(y_train)
        return y_train

    def inverse_transform_y(self, output):
        return self.y_encoder.inverse_transform(output)

    def get_n_output_node(self):
        return self.y_encoder.n_classes

    @property
    def metric(self):
        return Accuracy


class Image1DRegressor(Image1DSupervised):
    @property
    def loss(self):
        return regression_loss

    @property
    def metric(self):
        return MSE

    def get_n_output_node(self):
        return 1

    def transform_y(self, y_train):
        return y_train.flatten().reshape(len(y_train), 1)

    def inverse_transform_y(self, output):
        return output.flatten()


class PortableImage1DSupervised(PortableClass):
    def __init__(self, graph, data_transformer, y_encoder, metric, inverse_transform_y_method):
        """Initialize the instance.
        Args:
            graph: The graph form of the learned model
        """
        super().__init__(graph)
        self.data_transformer = data_transformer
        self.y_encoder = y_encoder
        self.metric = metric
        self.inverse_transform_y_method = inverse_transform_y_method

    def reshapeTo2D(self, X):
        #so data is already two-dimensional in that the x-dimension are training rows
        #and y-dimension are the 1D dimensional input data to the classifier.
        #In reality we are reshaping from 2D to 3D.

        len = X.shape[1]
        padToLen = len
        if padToLen<64: #needs to be at least 8x8 for 2D classification to work
            padToLen = 64
        elif padToLen % 8 > 0:
            padToLen += 8 - padToLen % 8 #make padToLen be divisble by 8.

        Xout = None
        if padToLen != len:
            X1 = np.empty((X.shape[0],padToLen))
            X1[:,0:len] = X
            Xout=X1
        else:
            Xout = copy.copy(X)
            
        Xout = Xout.reshape(Xout.shape[0],8,padToLen//8) #reshape to 2D image with all dimensions at least 8.
        Xout = Xout.reshape(Xout.shape + (1,))
        return Xout

    def predict(self, x_test):
        """Return predict results for the testing data.

        Args:
            x_test: An instance of numpy.ndarray containing the testing data.

        Returns:
            A numpy.ndarray containing the results.
        """
        if Constant.LIMIT_MEMORY:
            pass
        test_loader = self.data_transformer.transform_test(self.reshapeTo2D(np.array(x_test)))
        model = self.graph.produce_model()
        model.eval()

        outputs = []
        with torch.no_grad():
            for index, inputs in enumerate(test_loader):
                outputs.append(model(inputs).numpy())
        output = reduce(lambda x, y: np.concatenate((x, y)), outputs)
        return self.inverse_transform_y(output)

    def inverse_transform_y(self, output):
        return self.inverse_transform_y_method(output)

    def evaluate(self, x_test, y_test):
        """Return the accuracy score between predict value and `y_test`."""
        y_predict = self.predict(x_test)
        return self.metric().evaluate(y_test, y_predict)
