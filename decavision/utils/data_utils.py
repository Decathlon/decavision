import os
from random import shuffle
import sys
import tarfile
import urllib.request
import zipfile

import numpy as np
import PIL
from PIL import Image
import tensorflow as tf


def prepare_image(image_path, target_size, rescaling=255):
    """
    Load and convert image to numpy array to feed it to a neural network. Image is resized, converted to RGB
    and its pixels are normalized if required. An extra dimension is added to the array.

    Arguments:
        image_path (str): path to image to be converted
        target_size (tuple(int,int)): desired size for the image
        rescaling (int): divide all the pixels of the image by this number

    Returns:
        numpy array: processed image, with shape (1,target_size,3)
    """
    image = Image.open(image_path)
    # reshape the image
    image = image.resize(target_size, PIL.Image.BILINEAR).convert("RGB")
    # convert the image into a numpy array, and expend to a size 4 tensor
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # rescale the pixels to a 0-1 range
    image = image.astype(np.float32) / rescaling
    return image


def check_RGB(path, target_size=None):
    """
    Convert all images in a folder into RGB format and resize them if desired. Images that
    can't be opened are deleted. Folder must contain a subfolder for each category.

    Arguments:
        path (str): path to the image directory
        target_size (tuple(int,int)): if specified, images are resized to this size
    """

    classes = os.listdir(path)
    classes_paths = [os.path.abspath(os.path.join(path, i)) for i in classes]

    counter = 0
    for i in classes_paths:
        imgs = os.listdir(i)
        imgs_paths = [os.path.abspath(os.path.join(i, j)) for j in imgs]
        # Loop through all the images in the path
        for img in imgs_paths:
            # try to open it
            try:
                if target_size is not None:
                    jpg = Image.open(img).resize(target_size, PIL.Image.BILINEAR).convert('RGB')
                else:
                    jpg = Image.open(img).convert('RGB')
                jpg.save(str(img))
            except:
                # delete the file
                print('Deleting', img)
                os.remove(img)
            counter += 1
            if counter % 1000 == 1:
                print('Verified', counter, 'images')


def create_dir(path):
    """
    Check if directory exists and create it if it does not.

    Arguments:
        path (str): path to directory to create
    """
    if not os.path.exists(path):
        os.mkdir(path)


def split_train(path='data/image_dataset', split=0.1, with_test=False):
    """
    Separate images randomly into a training, a validation and potentially a test dataset.
    Images must be located in a folder called train, which contains a subfolder per category.
    Val and potentially test folders will be created amd images moved into it from train.

    Arguments:
        path (str): path to the image_dataset directory
        split (float): fraction of each category that we move to the validation (val) subdirectory
        with_test (bool): determine if one image of each category is moved to test dataset
    """

    # Create a val subdirectory
    create_dir(path + '/val')
    # Create a test subdirectory
    if with_test:
        create_dir(path + '/test')

    # Loop through all the categories in the train directory
    for i in os.listdir(path + '/train'):

        # Create the folder in the val subdirectory
        create_dir(path + '/val/' + i)

        # extract and shuffle all the images
        images = os.listdir(path + '/train/' + i)
        shuffle(images)

        # Move a fraction of the images to the val directory
        for j in range(int(split * len(images))):
            os.rename(path + '/train/' + i + '/' + images[j], path + '/val/' + i + '/' + images[j])

        # Move one of the images to the test directory
        if with_test:
            # Create the folder in the val subdirectory
            create_dir(path + '/test/' + i)

            for j in range(int(split * len(images)), 2*int(split * len(images))):
                os.rename(path + '/train/' + i + '/' + images[j], path + '/test/' + i + '/' + images[j])
    print('Training dataset has been split.')


def print_download_progress(count, block_size, total_size):
    """
    Function used for printing the download progress. Inspired by:
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/download.py
    """
    # Percentage completion.
    pct_complete = float(count * block_size) / total_size

    # Limit it because rounding errors may cause it to exceed 100%.
    pct_complete = min(1.0, pct_complete)

    # Status-message. Note the \r which means the line should overwrite itself.
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def download_dataset(download_dir='data/',
                     url='http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar'):
    """
    Download a dataset in format .zip, .tar, .tar.gz or .tgz and extract the data.
    Inspired by: https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/download.py

    Arguments:
        download_dir (str): folder to store the data
        url (str): location of the dataset on the internet
    """
    # Filename for saving the file downloaded from the internet.
    # Use the filename from the URL and add it to the download_dir.
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    # Check if the file already exists.
    # If it exists then we assume it has also been extracted,
    # otherwise we need to download and extract it now.
    if not os.path.exists(file_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Download the file from the internet.
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path,
                                                  reporthook=print_download_progress)

        print("\n Download finished. Extracting files. \n")

        if file_path.endswith(".zip"):
            # Unpack the zip-file.
            with zipfile.ZipFile(file=file_path, mode="r") as f:
                f.extractall(download_dir)

        elif file_path.endswith((".tar.gz", ".tgz")):
            # Unpack the tar-ball.
            with tarfile.open(name=file_path, mode="r:gz") as f:
                f.extractall(download_dir)

        elif file_path.endswith(".tar"):
            # Unpack tar file.
            with tarfile.open(name=file_path, mode="r") as f:
                f.extractall(download_dir)

        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")
