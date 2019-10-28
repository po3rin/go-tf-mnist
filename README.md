# go-tf-mnist

sample of Go & TensorFlow using model trained by Python.

## Quick Start

### Install

https://www.tensorflow.org/install/lang_go

### Prepare MNIST Data

download MNIST data from following url.
https://www.kaggle.com/scolianni/mnistasjpg

```bash
data
├── testSet
└── trainingSet
```

### Train

Because tensorflow/go requires version 1.13, model should be trained with v1.13.

```bash
# in train directory.
$ docker run -it -v $(pwd):/home python:3.7 bash
$ pip install -r requirements.txt # (tensorflow==1.13.1)
$ python version.py
1.13.1
$ python train.py
```

### Predict

```bash
# in predict directory
$ go run main.go < ./../train/data/testSet/img_1.jpg
```

### Reference

[yasuno0327/MyMNIST](https://github.com/yasuno0327/MyMNIST)
