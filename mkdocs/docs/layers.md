##WeightedAdd
Weighted Add class inherited from Add class
It's used to do add weights for data in Add layer
####Attributes
**weights**: backend variable

**one**: const 1.0

**kernel**: None

**_trainable_weights**: list that store weight

###__init__
Init Weighted add class

###call
Override call function in Add and return new weights

###compute_output_shape
Return output_shape

