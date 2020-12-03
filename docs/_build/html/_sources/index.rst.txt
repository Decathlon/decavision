:github_url: https://github.com/Decathlon/decavision.git

.. decavision documentation master file, created by
   sphinx-quickstart on Fri Jul 24 09:14:44 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DecaVision's documentation!
========================================================

This library contains the methods required to build an image classification neural network using transfer learning.

The module can be used to prepare a dataset of images for training, train a classification model 
built on top of various pretrained models, optimize model hyperparameters using scikit-optimize library and evaluate 
the accuracy on a test set. The library capitalizes on the concepts of data augmentation, fine tuning and 
hyperparameter optimization, to achieve high accuracy given small sets of training images.

This library has a few distinguishing features. It is specifically designed to work with Google Colab notebooks and leverage their TPUs, to seamlessly transition from quick iterations of modeling approaches to large-scale training. Functionalities of hyperparmaters tuning and progressive learning are included and easily integrated in the pipeline to reach higher accuracy. Finally, state-of-the art efficientnet transfer learning models and Tensorflow 2 functionalities are considered to provide high efficiency. 

A great way of explaining the library is using an example notebook, which you can find `here <https://colab.research.google.com/drive/1s9pnNotdPoHI4n_7OZl4zwEP9rccJ3f-?usp=sharing/>`_. 

Installation
=============

This library works with python 3.6 and above and it is based on the following dependencies:

- efficientnet
- tensorflow 2
- matplotlib
- scikit-optimize
- pillow
- numpy
- pandas
- scikit-learn
- augmentor
- dill
- google-cloud-storage
- pyunpack
- patool
- seaborn

This library is available through the Python Package Installer (PyPI) by typing:

``pip install decavision``

All the dependencies are installed along with the library, so it is safer to perform the installation in a fresh virtual environment.

Contents
========

This documentation is separated in two distinct parts. The first one explains what the functions made available to you do. The second part
shows examples of how to use the code explicitely.

.. toctree::
   :maxdepth: 3
   :caption: Functions

   data
   train
   testing
   utils
   

.. toctree::
   :maxdepth: 2
   :caption: Examples

   data_example
   train_example

Roadmap
=======

Follow the `Sport Vision API <https://developers.decathlon.com/products/sport-vision/docs>`_ and our blog at `<https://medium.com/decathlondevelopers/>`_ for updates.

Support
========

Pull requests to the library are welcomed.  If you have any problem, question or suggestion regarding this library, don't hesitate to create an issue or contact us at aicanada@decathlon.net.
