Train and optimize model
==========================

This code example shows how to use this library to train an image classification model from scratch, using a dataset saved in tfrecords format.
You will also perform an optimization of the hyperparameters of the model to achieve the best accuracy possible.

Training the classification model
----------------------------------

To train an image classification model, you need to have your training and validation data saved in tfrecords format, as is explained in 
:doc:`data_example`. We will continue working with this example. The data is in a directory with the following structure::

  data/
    tfrecords/
      classes.csv
      train/
        tfrecords_train.tfrec
        ...
      val/
        tfrecords_val.tfrec
        ...

The training is then done with the following code::

  classifier = decavision.model_training.tfrecords_image_classifier.ImageClassifier(tfrecords_folder='data/tfrecords', batch_size=16, transfer_model='B3')
  classifier.fit()
 
You can decide the transfer model between Xception, Inception_Resnet, Resnet and B0, B3, B5 or B7 (all EffecientNets). Their respective 
sizes and performance metrics can be found in the keras `documentation <https://keras.io/api/applications/>`_. Also, note that on the fly data augmentation is done by default so if you already generated new images manually be sure to set augment to False.

The parameters that can be specified when training are:

* hyperparameters (learning_rate, learning_rate_fine_tuning, epochs, hidden_size, dropout, activation, l2_lambda)
* the option to save the model after saving (save_model for h5, export_model for pb)
* callbacks (min_accuracy for earlystopping, logs for tensorboard)
* verbose

This function trains an extra layer on top of the pretrained model and then fine tunes a few of the last layers of the pretrained model.

Saving and exporting the keras model
-------------------------------------

Once you are satisfied with your results, you can save your trained model using any of the following two methods:

* Specify either the save_model (for .h5 format) or export_model (for .pb format, to use with tfserving) argument when training. 
  The value of the argument will be the name of the file saved after training.
* After training, run the following code::

    classifier.model.save(filename)

  If the filename has the extension .h5 the model will be saved in that format. Otherwise it will be saved in .pb. With a TPU, 
  it is only possible to export a model to a google cloud bucket.


Optimization of the hyperparameters
------------------------------------

There is no specific rule to select the values of the many hyperparameters so you have to try a variation and find the best. 
This can be done automatically using the following code::

  classifier = decavision.model_training.tfrecords_image_classifier.ImageClassifier(tfrecords_folder='tfrecords', batch_size=16, transfer_model='B3')
  classifier.hyperparameter_optimization(num_iterations=25, n_random_starts=10)

This performs a series of training with various combinations of hyperparameters to find the best model. The most important parameters of 
this method determine how many random tries to start with (n_random_starts) and how many total tries to make (num_iterations). After 
the algorithm is done with the random tries, it starts to learn from the past tries to find better combinations. This is done using 
scikit-optimize. Every iteration, a checkpoint.pkl file is saved and uploaded to your drive so you don't lose your progress in 
Google Colab (if you are using it). If you want to restart from a previous checkpoint.pkl, the file must be in your working directory.
During optimization, the results of all the tries are saved in a log file for future reference.
