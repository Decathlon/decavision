Build a dataset from scratch
=============================

This code example shows how you can use this library to prepare a dataset of images for training. You can 
perform data augmentation on your images and transform the dataset into tfrecords to be used to train a classification model. 
The data can come from any source and the only requirement is that the images are saved in separate folders for each class.


Splitting a set of images into a training and a validation set
----------------------------------------------------------------

If you have a single set of images, but would like to split it into a training and a validation set, first place your images in 
a train folder located in 'data/image_dataset' for this example. Using hockey and soccer players as examples
of classes, your dataset should be organized in the directory as follows::

  data/
    image_dataset/
      train/
        hockey_player/
          hockey_player_1.jpg
          hockey_player_2.jpg
          ...
        soccer_player/
          soccer_player_1.jpg
          soccer_player_2.jpg
          ...        

Then, you can run the following code::

  decavision.utils.data_utils.split_train(path='data/image_dataset', split=0.2)

This will go through all the classes in the training set, randomly pick a fraction (*split* argument) of the images, and 
move them from the train/ directory to a newly created val/ directory.


Make data augmentation on training images
-----------------------------------------

You can increase the training dataset by using data augmentation. You can choose between all these options : distortion,
flip_horizontal, flip_vertical, random_crop, random_erasing, rotate, resize, skew, shear, brightness, contrast, color. All the images 
in the specified folder will have a combination of the desired augmentations. Then you need to specify a desired number of images for all the classes, and it will generate new images to get as close as possible 
to that number::

  augmentor = decavision.dataset_preparation.data_augmentation.DataAugmentor(path='data/image_dataset/train', distortion=True, flip_horizontal=True, flip_vertical=True)
  augmentor.generate_images(100)

If you don't want to generate new images there is an option to do data augmentation during training by modifying the training data online.


Generate Tfrecord files
-----------------------

Once this is done you can generate Tfrecords, which is hardly recommended if you want to use TPU, or just to get better performance 
from your GPU. It is recommended that you create 100 Mb file using multiple shards. Also, you can save the generated files in a 
GCS Bucket (gs://) as the output folder (only if you are on colab and have authenticated your account)::

  generator = decavision_dataset_preparation.generate_tfrecords.TfrecordsGenerator()
  generator.convert_image_folder(img_folder='data/image_dataset/train', output_folder='data/tfrecords/train')

You need to use this code once for your train, val (and test folders if you have one). This will delete the old records in the folders and create 
a csv file with the names of your classes.

The images are not resized by default because different models use different sizes during training. There is an option to specify a target size if desired.

  
  
