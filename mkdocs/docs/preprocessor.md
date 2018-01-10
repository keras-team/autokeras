##OneHotEncoder
A class that can format data
This class provide ways to transform data's classification into vector
####Attributes
**data**: the input data

**n_classes**: the number of classification

**labels**: the number of label

**label_to_vec**: mapping from label to vector

**int_to_label**: mapping from int to label

###__init__
Init OneHotEncoder

###fit
Create mapping from label to vector, and vector to label

###transform
Get vector for every element in the data array

###inverse_transform
Get label for every element in data

