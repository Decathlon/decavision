Improve a model using unlabelled images
========================================

This code example shows how to use this library to exploit unlabelled data to increase the performance of an image
classifier. The main procedure consists in first training a classifier using the labelled data (or using an already
trained one). Then this model is used to make predictions for the unlabelled data, which are used as pseudo labels to
create a new dataset. This larger dataset is then used to train a larger and more performant classifier.


Preliminary steps
------------------

The first ingredient necessary to use semi-supervised learning is a labelled dataset, exactly as described in
:doc:`data_example`, saved at 'data/image_dataset'. This dataset is used as in :doc:`train_example` to train the best
possible model, which is called in this context the *teacher* and is saved as 'model.h5'. Of course in the context
where you want to improve a model, you skip this step and use your own existing model.

The other ingredient is a dataset of unlabelled images. This is ideally (but not necessarily) composed of relevant
images to the problem that are not split into categories. The images are saved in a single folder called 'data/unlabeled'.


Generating pseudo labels
--------------------------

The most important step in semi-supervised learning consists in labelling the unlabelled data, which is done using the
following code::

  generator = decavision.dataset_preparation.generate_pseudolabels.PseudoLabelGenerator()
  generator.generate_pseudolabel_data()

This uses the teacher model to predict the label of each unlabelled image. These predictions are then saved in a csv file at 'outputs/data.csv'.

Since the unlabelled images can come from different places, it is important to make sure that bad images are not used
further. To ensure data quality, it can be helpful to plot the distribution of the predictions using::

  generator.plot_confidence_scores(per_class=False)

A chart will be saved in the 'outputs' folder with the highest probability predicted for each image. If some probabilities
are too low, it means that the image is so bad that the teacher model does not recognize it. Such image should ideally
be discarded. This chart thus helps pick the threshold variable that will be used to create the new dataset with only relevant images.

The final step is to use the predictions along with the threshold (if one is used) to distribute the unlabelled images
into classes. This pseudo labelled dataset is then combined with the original training data to make a larger dataset::

  generator.move_unlabeled_images(threshold=None)

Note that the original datasets are kept intact. The larger dataset is made from copies of the images.

All these steps can also be done directly with the following code::

  generator.generate_pseudolabel_data(plot_confidences=True, threshold=None, move_images=True)



Training the student model
----------------------------

With the larger dataset in hand the last step is to train a new and improved model, the *student* model, using the same
method that was used for the teacher model. There are only a few points that are important to keep in mind to achieve
the best model possible:

- Make sure that you use data augmentation when training the student model.
- Use a larger model. For example if you used EfficientNet B3 for the teacher, try B5 for the student.