# Neural Architecture Search

To help the researchers to do experiments on neural architecture search (NAS),
we have implemented several baseline methods using the Auto-Keras framework.
The implementations are easy since only the core part of the search algorithm is needed.
All other parts of NAS (e.g. data structures for storing neural architectures, training of the neural networks)
are done by the Auto-Keras framework.

## Why implement NAS papers in Auto-Keras?

The NAS papers usually evaluate their work with the same dataset (e.g. CIFAR10),
but they are not directly comparable because of the data preparation and training process are different,
the influence of which are significant enough to change the rankings of these NAS methods.

We have implemented some of the NAS methods in the framework.
More state-of-the-art methods are in progress.
There are three advantages of implementing the NAS methods in Auto-Keras.
First, it fairly compares the NAS methods independent from other factors
(e.g. the choice of optimizer, data augmentation).
Second, researchers can easily change the experiment datasets used for NAS.
Many of the currently available NAS implementations couple too much with the dataset used,
which makes it hard to replace the original dataset with a new one.
Third, it saves the effort of finding and running code from different sources.
Different code may have different requirements of dependencies and environments,
which may conflict with each other.

## Baseline methods implemented

Description of each baseline method.

## How to run the baseline methods?

Code example containing two parts: data preparation, search.
Should call CnnModule.


## How to implement your own search?

You are welcome to implement your own method for NAS in our framework.
If it works well, we are happy to merge it into our repo.

