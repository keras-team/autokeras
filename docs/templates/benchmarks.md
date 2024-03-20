# Benchmarks

We track the performance of the latest AutoKeras release on the benchmark datasets.
Tested on a single NVIDIA Tesla V100 GPU.

| Name | API | Metric | Results | GPU Days |
| - | - | - | - | - |
| [MNIST](http://yann.lecun.com/exdb/mnist/)  | ImageClassifier| Accuracy | 99.04% | 0.51 |
| [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)   | ImageClassifier| Accuracy | 97.10% | 1.8 |
| [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/)  | TextClassifier | Accuracy | 93.93% | 1.2 |