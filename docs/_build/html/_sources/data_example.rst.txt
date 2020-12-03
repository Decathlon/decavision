Build a dataset from scratch
=============================

This code example shows how you can use this library to build a dataset from scratch by downloading images from Google. You can then
perform data augmentation on your images and transform the dataset into tfrecords to be used to train a classification model.


Building training, validation and test sets of images
------------------------------------------------------

To build a set of images to train your neural network, first create a folder to store your images. The default name that the library uses
for this folder is "google_data", but you can use whatever you want. Then create in this flder a csv file called "searchterms.csv" 
to indicate the categories you want to classify, the search terms from which you would like to build your sets of images, along 
with the number of images you want to extract from the Google search. For instance, let's assume you want to build a training and 
a validation set to distinguish a hockey player from a soccer player. As an example, the searchterms.csv file could look as follows:

+------------------+-------------+---------------+
| search_term      | number_imgs | category      |
+==================+=============+===============+
| hockey player    |     150     | hockey_player |
+------------------+-------------+---------------+
| soccer player    |     150     | soccer_player |
+------------------+-------------+---------------+
| joueur de hockey |     20      | hockey_player |
+------------------+-------------+---------------+
| joueur de soccer |     20      | soccer_player |
+------------------+-------------+---------------+

In this example, we capitalize on searching equivalent terms in different languages (in this case English, 
*hockey player*, and French, *joueur de hockey*) to extract different images.

Then, you can run the following code::

  extractor = decathlonian.dataset_preparation.extract_images.ExtractImages()
  extractor.run()

This will extract the desired number of images for each search term, and store them in a *google_data/image_dataset* directory organized as follows::

  google_data/
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
        

Once this set is built, **make sure you take a quick look at the images**, to cleanup the dataset and remove the images not relevant to the 
classification problem at hand. Note that you can have multiple search terms for a given class of images, and that there is no limit to 
the number of different categories you want your model to classify. Note that if you want to use this functionality, make sure to 
have your `chromedriver executable <http://chromedriver.chromium.org/>`_ in the data directory. Also note that this functionnality doesn't
work on colab because the web browser will not be able to start.


Spliting a set of images into a training and a validation set
-------------------------------------------------------------

If you have a single set of images, but would like to split it into a training and a validation set, first place your images in 
a train folder. Keeping the hockey and soccer players example, your dataset should be organized in the directory as follows::

  google_data/
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

  decathlonian.utils.data_utils.split_train(path='google_data/image_dataset', split=0.2)

This will go through all the classes in the training set, randomly pick a fraction (*split* argument) of the images, and 
move them from the train/ directory to a newly created val/ directory.


Make data augmentation on training images
-----------------------------------------

You can increase the training dataset by using data augmentation. You can choose between all these options : distortion,
flip_horizontal, flip_vertical, random_crop, random_erasing, rotate, resize, skew, shear, brightness, contrast, color. All the images 
in the specified folder will have a combination of the desired augmentations. Then you need to specify a desired number of images for all the classes, and it will generate new images to get as close as possible 
to that number::

  augmentor = decathlonian.dataset_preparation.data_augmentation.DataAugmentor(path='google_data/image_dataset/train', distortion=True, flip_horizontal=True, flip_vertical=True)
  augmentor.generate_images(100)


Generate Tfrecord files
-----------------------

Once this is done you can generate Tfrecords, which is hardly recommended if you want to use TPU, or just to get better performance 
from your GPU. It is recommended that you create 100 Mb file using multiple shards. Also, you can save the generated files in a 
GCS Bucket (gs://) as the output folder (only if you are on colab and have authenticated your account)::

  generator = decathlonian_dataset_preparation.generate_tfrecords.TfrecordsGenerator()
  generator.convert_image_folder(img_folder='google_data/image_dataset/train', output_folder='google_data/tfrecords/train')

You need to use this code once for your train, val and test folders. This will delete the old records in the folders and create 
a csv file with the names of your classes.

  
  
