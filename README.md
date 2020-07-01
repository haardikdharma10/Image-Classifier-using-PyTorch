<h1 align="center">Image Classifier using PyTorch</h1>

<div align= "center">
  <h4>Image Classification system built with PyTorch using Deep Learning concepts in order to recognize different species of flowers.</h4>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/haardikdharma10/Image-Classifier-using-PyTorch/issues)
[![Forks](https://img.shields.io/github/forks/haardikdharma10/Image-Classifier-using-PyTorch.svg?logo=github)](https://github.com/haardikdharma10/Image-Classifier-using-PyTorch/network/members)
[![Stargazers](https://img.shields.io/github/stars/haardikdharma10/Image-Classifier-using-PyTorch.svg?logo=github)](https://github.com/haardikdharma10/Image-Classifier-using-PyTorch/stargazers)
[![Issues](https://img.shields.io/github/issues/haardikdharma10/Image-Classifier-using-PyTorch.svg?logo=github)](https://github.com/haardikdharma10/Image-Classifier-using-PyTorch/issues)
[![MIT License](https://img.shields.io/github/license/haardikdharma10/Image-Classifier-using-PyTorch.svg?style=flat-square)](https://github.com/haardikdharma10/Image-Classifier-using-PyTorch/blob/master/LICENSE)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

![Test on a random image from the dataset](https://github.com/haardikdharma10/Image-Classifier-using-PyTorch/blob/master/assets/test1.png)

## Motivation
Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smartphone app. To do this, you'd use a deep learning model trained on hundreds and thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, I have trained an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice, you'd train this classifier, then export it for use in your application. We'll be using this [dataset](https://github.com/haardikdharma10/Image-Classifier-using-PyTorch/blob/master/flowers) of more than 8000 images and 102 flower categories.

When you've completed this project, you'll have an application that can be trained on any set of labelled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset.

## Framework/Model used
- [PyTorch](https://pytorch.org/)
- [VGG](https://arxiv.org/abs/1409.1556)

## Prerequisites

All the dependencies and required libraries are included in the file [requirements.txt](https://github.com/haardikdharma10/Image-Classifier-using-PyTorch/blob/master/requirements.txt)

## Installation
1. Clone the repo
```
$ git clone https://github.com/haardikdharma10/Image-Classifier-using-PyTorch.git
```

2. Change your directory to the cloned repo and create a Python virtual environment named 'testenv'
```
$ mkvirtualenv testenv
```

3. Now, run the following command in your Terminal/Command Prompt to install the libraries required
```
$ pip3 install -r requirements.txt
```
## GPU
As the network makes use of a sophisticated deep convolutional neural network, the training process is impossible to be done by a common laptop. In order to train your models to your local machine you have three options

1. **Cuda** - If you have an NVIDIA GPU then you can install CUDA from [here](https://developer.nvidia.com/cuda-downloads). With Cuda you will be able to train your model however the process will still be time consuming
2. **Cloud Services** - There are many cloud services that let you train your models like [AWS](https://aws.amazon.com/) and [Google Cloud](https://cloud.google.com/)
3. **Coogle Colab** - [Google Colab](https://colab.research.google.com/) gives you free access to a tesla K80 GPU for 12 hours at a time. Once 12 hours have ellapsed you can just reload and continue! The only limitation is that you have to upload the data to Google Drive and if the dataset is massive you may run out of space.

However, once a model is trained then a normal CPU can be used for the predict.py file and you will have an answer within some seconds.

## Authors
* **Haardik Dharma** - Initial Work
* **Udacity** - Final project of the 'AI with Python Nanodegree'

## Credits
* [https://www.udacity.com/course/ai-programming-python-nanodegree--nd089](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089)
* [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/haardikdharma10/Image-Classifier-using-PyTorch/blob/master/LICENSE) file for details. 
![alt text](https://github.com/haardikdharma10/Image-Classifier-using-PyTorch/blob/master/assets/Certificate.jpg)


