Preparing datasets
=====================

The following code details the functions and classes that are available to build a dataset of images and prepare it for
training. Functions to label unlabelled data to use for semi-supervised learning are also presented.

Examples of how to use these functions can be found in :doc:`data_example` and in :doc:`ssl_example`.

Data augmentation
---------------------

.. automodule:: decavision.dataset_preparation.data_augmentation
   :members:

Make tfrecords
---------------------

.. automodule:: decavision.dataset_preparation.generate_tfrecords
   :members:

Generate pseudo labels for unlabelled data
-------------------------------------------

.. automodule:: decavision.dataset_preparation.generate_pseudolabels
   :members:
