# CnnModule tutorial

`CnnModule` in `net_module.py` is a child class of `Networkmodule`. It can generates neural architecture with basic cnn modules
and the ResNet module. 

### Examples
Normally, there's two place to call the CnnModule, one is call `CnnModule.fit` while the other is `CnnModule.final_fit`.

For example, in a image classification class `ImageClassifier`, one can initialize the cnn module as:

```python
self.cnn = CnnModule(loss, metric, searcher_args, path, verbose)
```
Where:
* `loss` and `metric` determines by the type of training model(classification or regression or others)
* `search_args` can be referred in `search.py`
* `path` is the path to store the whole searching process and generated model.
* `verbose` is a boolean. Setting it to true prints to stdout.

Then, for the searching part, one can call:
```python
self.cnn.fit(n_output_node, input_shape, train_data, test_data, time_limit=24 * 60 * 60)
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
self.cnn.final_fit(train_data, test_data, trainer_args=None, retrain=False)
```
where:
* train_data: A DataLoader instance representing the training data.
* test_data: A DataLoader instance representing the testing data.
* trainer_args: A dictionary containing the parameters of the ModelTrainer constructor.
* retrain: A boolean of whether reinitialize the weights of the model.