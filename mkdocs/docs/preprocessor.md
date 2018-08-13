##OneHotEncoder
A class that can format data.
This class provides ways to transform data's classification label into vector.
#####Attributes
* **data**: The input data

* **n_classes**: The number of classes in the classification problem.

* **labels**: The number of labels.

* **label_to_vec**: Mapping from label to vector.

* **int_to_label**: Mapping from int to label.

###__init__
Initialize a OneHotEncoder

###fit
Create mapping from label to vector, and vector to label.

###transform
Get vector for every element in the data array.

###inverse_transform
Get label for every element in data.

