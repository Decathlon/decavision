Train models
=====================

The following code details the functions and classes that are available to train an image classification model and opimize its 
hyperparameters. Progressive learning can also be performed to add new classes to a model that was already trained using 
the library without losing all the information learned.

An example of how to use these functions can be found in :doc:`train_example`.


Image classification
---------------------

.. automodule:: decavision.model_training.tfrecords_image_classifier
   :members:

Progressive learning
---------------------

.. automodule:: decavision.model_training.progressive_learning
   :members:
