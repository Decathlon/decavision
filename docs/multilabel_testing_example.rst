Testing Multilabel image classification model
===============================================

This code example shows how to use this library to test a multilabel image classification model from scratch, using an already trained model. 
You can also create a movie from the classified images with predicted labels on them. 

Evaluating the multilabel classification model
------------------------------------------------

You can use this library to evaluate the trained multilabel image classification model with different tools. The simplest test to do is evaluating 
the f1-score on different datasets where you could just place the images in say a **val/** directory. The data is in a directory with the following structure::

  image_dataset/
      classes.json
      train/
          sun_aabeeufygtjcsego.jpg  
          ...
      val/
          sun_aaabhshfqutsklcz.jpg
           ...

First, the list of classes or categories should be extracted using the following code::

    categories = decavision.utils.utils.load_classes(gs_folder)

Then the evaluation of the model is done using the following code::
    
    tester = decavision.model_testing.testing.ModelTesterMultilabel(model=model_name, categories=categories)
    tester.evaluate(path="image_dataset/val", json_file="image_dataset/classes.json")


Plot & Save classified images 
-------------------------------

You can also explicitly look at the classified images with predictions on the fly.
To do so use the following function::

  tester.classify_images(path="image_dataset/val", json_file="image_dataset/classes.json", plot=True, save_img=True)

In order to save images, you will need to specify :code:`plot=True` and :code:`save_img=True`. You will not be able to save
images without plotting, this will be updated in the next version of the library. 

.. admonition:: Note!

  Set :code:`plot=True, save_img=True` to save classified images. 

Create a movie from classified images
--------------------------------------

There are two ways to create a movie from classified images: you can directly run the following code by setting :code:`classify_images=True` to 
make predictions on new images -> save classified images to a folder and create a movie from them:: 

  tester.create_movie(path="image_dataset/val", classify_images=True)

If you already have classified saved images in a folder, you can set :code:`classify_images=False` and pass an optional argument which will be the path 
to the classified saved images directory :code:`image_folder=classified_image_path`. Assume the classified images are saved under **image_dataset/classified_images/**, then::
  
  tester.create_movie(path=path, classify_images=False, image_folder="image_dataset/classified_images")


Generate Classification report and Confusion matrix
-----------------------------------------------------

Finally, you can also generate a confusion matrix and classification report using the function::

  tester.generate_metrics(path="image_dataset/val", json_file)
