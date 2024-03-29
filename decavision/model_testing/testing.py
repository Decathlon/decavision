import glob
import math
import os
import random
import json
import cv2

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model

from decavision.utils import data_utils
from decavision.utils import utils
from decavision.utils.training_utils import f1_score


class ModelTester:
    """
    Class to use a trained image classification model on some images. Using this
    when working with a TPU disables eager execution.

    Arguments:
        model (str): path to trained model
    """

    def __init__(self, model):
        use_tpu, use_gpu = utils.check_PU()
        if use_tpu:
            # necessary because keras generators don'T work with TPUs...
            tf.compat.v1.disable_eager_execution()
        try:
            self.model = load_model(
                model,
                custom_objects={"_f1_score": f1_score, "f1_score": f1_score})
            # efficientnets have the scaling included in them so no need to
            # rescale the images when loading
            if self.model.name[0] in ['B', 'V']:
                self.rescaling = 1
            else:
                self.rescaling = 255
            print('Model loaded correctly')
        except Exception as e:
            print('There was a problem when trying to load your model: {}'.format(e))

        self.input_shape = self.model.input_shape[1:3]

    def _load_dataset(self, path):
        """
        Load dataset into a keras generator. Images must be contained in separate
        folders for each class.

        Arguments:
            path (str): location of the dataset

        Returns:
            generator: images plus information about them (labels, paths, etc)
        """
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1 / self.rescaling)
        generator = datagen.flow_from_directory(directory=path,
                                                target_size=self.input_shape,
                                                shuffle=False,
                                                interpolation='bilinear',
                                                color_mode='rgb',
                                                class_mode='categorical',
                                                batch_size=1)
        return generator

    def confusion_matrix(self, path, normalize=None):
        """
        Compute and plot the confusion matrix resulting from predictions on a dataset of images.
        Images must be located in separate folders for each class.

        Arguments:
            path (str): location of the images
            normalize ('true', 'pred', 'all' or None): normalizes confusion matrix over the true (rows), predicted (columns) conditions or
                all the population. If None, confusion matrix will not be normalized.
        """
        generator = self._load_dataset(path)
        cls_true = generator.classes
        labels = list(generator.class_indices.keys())
        cls_pred = self.model.predict(generator)
        cls_pred = np.argmax(cls_pred, axis=1)
        print('Labels loaded')

        # Calculate the confusion matrix.
        cm = confusion_matrix(y_true=cls_true,  # True class for test-set.
                              y_pred=cls_pred,  # Predicted class.
                              normalize=normalize)

        # Print the confusion matrix
        ax = plt.subplot()
        sn.heatmap(cm, annot=True, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)

    def _plot_images(self, images, categories, cls_true, cls_pred=None, smooth=True):
        """
        Plot images along with their true and optionally predicted labels.
        Inspired by https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/10_Fine-Tuning.ipynb.

        Arguments:
            images (list[numpy arrays]): list of images to plot as arrays
            categories (List[str]): list of categories that model predicts
            cls_true (np.array[int]): true labels of the images
            cls_pred (np.array[int]): predicted labels of the images
            smooth (bool): smooth out images or not when plotting
        """
        assert len(images) == len(cls_true)
        num_images = len(images)

        # Create figure with sub-plots.
        if math.sqrt(num_images).is_integer():
            nrows = ncols = int(math.sqrt(num_images))
        else:
            for i in reversed(range(math.ceil(math.sqrt(num_images)))):
                if not num_images % i:
                    nrows = int(num_images / i)
                    ncols = int(i)
                    break
        fig, axes = plt.subplots(nrows, ncols)

        # Adjust vertical spacing.
        if cls_pred:
            hspace = 0.6
        else:
            hspace = 0.3
        fig.subplots_adjust(hspace=hspace, wspace=0.3)

        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        for i, ax in enumerate(axes.flat):
            # Plot image.
            ax.imshow(images[i], interpolation=interpolation)
            # Name of the true class.
            cls_true_name = categories[cls_true[i]]
            # Show true and predicted classes.
            if cls_pred:
                # Name of the predicted class.
                cls_pred_name = categories[cls_pred[i]]
                xlabel = "True: {0}\nPred: {1}".format(
                    cls_true_name, cls_pred_name)
            else:
                xlabel = "True: {0}".format(cls_true_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.tight_layout()
        plt.show()

    def plot_errors(self, path, num_pictures=9):
        """
        Plot images that were not classified correctly by the model. Images to test must
        be located in separate folders for each class, for example a validation dataset.

        Arguments:
            path (str): location of the images
            num_pictures (int): maximum number of errors to show
        """
        generator = self._load_dataset(path)
        cls_true = generator.classes
        image_paths = generator.filepaths
        labels = list(generator.class_indices.keys())
        cls_pred = self.model.predict(generator)
        cls_pred = np.argmax(cls_pred, axis=1)
        print('Labels loaded')

        # get all errors index
        errors = []
        for i in range(len(cls_pred)):
            if cls_pred[i] != cls_true[i]:
                errors.append(i)

        # Load images randomly picked
        num_pictures = min(num_pictures, len(errors))
        random_errors = sorted(random.sample(errors, num_pictures))

        # Plot the images we have loaded and their corresponding classes.
        self._plot_images(
            images=[data_utils.prepare_image(image_paths[i], self.input_shape)[
                0] for i in random_errors],
            categories=labels,
            cls_true=[cls_true[i] for i in random_errors],
            cls_pred=[cls_pred[i] for i in random_errors]
        )

    def classify_images(self, image_path, categories, plot=True):
        """
        Classify images located directly in a folder. Plots the images and the first three predictions.

        Arguments:
            image_path (str): location of the images
            categories (list[str]): list of potential categories that the model can return
            plot (bool): plot or not the images, if False, only results are printed
        """
        images = glob.glob(os.path.join(image_path, '*.jpg'))
        for image_path in images:
            # prepare the image
            image_tensor = data_utils.prepare_image(
                image_path, self.input_shape, self.rescaling)
            # make and decode the prediction
            result = self.model.predict(image_tensor)[0]
            # print image and top predictions
            top_pred = np.argsort(result)[::-1][:3]
            # Name of the true class.
            cls_pred_name = np.array(categories)[top_pred]
            cls_pred_perc = result[top_pred] * 100
            if plot:
                if self.model.name[0] not in ["B", "V"]:
                    plt.imshow(image_tensor[0], interpolation='nearest')
                else:
                    plt.imshow(image_tensor[0].astype('uint8'), interpolation='nearest')
                xlabel = 'Prediction :\n'
                for (x, y) in zip(cls_pred_name, cls_pred_perc):
                    xlabel += '{0}, {1:.2f}%\n'.format(x, y)
                plt.xlabel(xlabel)
                plt.xticks([])
                plt.yticks([])
                plt.show()
            else:
                print('\nImage: ', image_path)
                for i in range(len(top_pred)):
                    print('Prediction: {} (probability {}%)'.format(
                        cls_pred_name[i], round(cls_pred_perc[i])))

    def evaluate(self, path):
        """
        Calculate the accuracy of the model on a dataset of images. The images must be
        in separate folders for each class.

        Arguments:
            path (str): location of the dataset
        """
        generator = self._load_dataset(path)
        results = self.model.evaluate(generator)
        print('Accuracy of', results[1] * 100, '%')

    def generate_classification_report(self, path):
        """
        Computes classification report resulting from predictions on a dataset of images
        and prints the results. Images must be located in separate folders for each class. The report shows average accuracy, precision, recall and f1-scores. Precision, recall and f1-scores are also computed for each class.

        Arguments:
            path (str): location of the images
        """
        generator = self._load_dataset(path)
        cls_true = generator.classes
        labels = list(generator.class_indices.keys())
        cls_pred = self.model.predict(generator)
        cls_pred = np.argmax(cls_pred, axis=1)
        print('Labels loaded')
        # Show classification report
        print(classification_report(
            cls_true, cls_pred, target_names=labels, digits=4))


class ModelTesterMultilabel:
    """
    Class to use a trained multilabel image classification model on some images. Using this
    when working with a TPU disables eager execution.

    Arguments:
        model (str): path to trained model
        categories (list[str]): list of potential categories that the model can return
    """

    def __init__(self, model, categories):
        use_tpu, use_gpu = utils.check_PU()
        if use_tpu:
            # necessary because keras generators don'T work with TPUs...
            tf.compat.v1.disable_eager_execution()
        try:
            self.model = load_model(model, custom_objects={
                                    "_f1_score": f1_score, "f1_score": f1_score})
            # efficientnets have the scaling included in them so no need to
            # rescale the images when loading
            if self.model.name[0] in ['B', 'V']:
                self.rescaling = 1
            else:
                self.rescaling = 255
            print('Model loaded correctly')
        except Exception as e:
            print('There was a problem when trying to load your model: {}'.format(e))

        self.input_shape = self.model.input_shape[1:3]
        self.categories = categories
        print("Model name: ", self.model.name)

    def _load_dataset(self, path, json_file):
        """
        Load dataset into a keras generator. Images must be contained in single
        folder. A dataframe is required with 2 columns: original
        image name and image labels as a list separated by comma. i.e.

        filenames                |    labels
        -------------------------|-----------------
        sun_abifwwwgjnomvfda.jpg | [asphalt,clouds,natural_light,man-made,open_area,far-away_horizon,sky,barn,airfield]
        sun_aciuvbhvntkgdhhk.jpg | [grass,asphalt,natural_light,natural,man-made,open_area,far-away_horizon,sky,dirt/soil,airfield]

        Arguments:
            path (str): location of the dataset
            json_file (str): path to json file containing image ids and their associated labels

        Returns:
            generator: images plus information about them (labels, paths, etc)
        """
        data = []
        with open(json_file, 'r') as file:
            values = json.load(file)
            for f, l in values.items():
                data.append({"filenames": f, "labels": l})

        df = pd.DataFrame(data)

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / self.rescaling)
        generator = datagen.flow_from_dataframe(directory=path,
                                                dataframe=df,
                                                x_col="filenames",
                                                y_col="labels",
                                                target_size=self.input_shape,
                                                shuffle=False,
                                                color_mode='rgb',
                                                class_mode='categorical',
                                                classes=self.categories,
                                                batch_size=1)
        return generator

    def classify_images(self, path, json_file, threshold=0.5, plot=False, save_img=False):
        """
        Classify images located directly in a folder. Plots and saves the images with the specified threshold.

        Arguments:
            path (str): location of the images
            json_file (str): path to json file containing image ids and their associated labels
            threshold (int): threshold for prediction (default to 0.5)
            plot (bool): plot or not the images, if False, only results are printed
            save_img (bool): save classified images or not in a new folder. To save images, you will need to set plot=True, otherwise no images will be saved. 
        """
        with open(json_file, 'r') as file:
            values = json.load(file)

        images = glob.glob(os.path.join(path, '*.jpg'))
        for image_path in images:
            # prepare the image
            image_tensor = data_utils.prepare_image(image_path, self.input_shape, self.rescaling)
            # make and decode the prediction
            result = self.model.predict(image_tensor)[0]
            top_pred = result > threshold
            # Name of the true class.
            cls_pred_name = np.array(self.categories)[top_pred]
            cls_pred_perc = result[top_pred] * 100
            cls_true = values[os.path.basename(image_path)]

            if plot:
                fig, ax = plt.subplots()
                if self.model.name[0] not in ["B", "V"]:
                    ax.imshow(image_tensor[0], interpolation='nearest')
                else:
                    ax.imshow(image_tensor[0].astype('uint8'), interpolation='nearest')
                xlabel = 'Prediction :\n'
                for (x, y) in zip(cls_pred_name, cls_pred_perc):
                    xlabel += '{0}, {1:.2f}%\n'.format(x, y)
                ax.set_xlabel(xlabel)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.tight_layout()
                if save_img:
                    data_utils.create_dir("classified_images")
                    fig.savefig("classified_images/" + os.path.basename(image_path))
            else:
                print('\nImage: ', image_path)
                print("True label:", cls_true)
                for i in range(len(cls_pred_perc)):
                    print('Prediction: {} (probability {}%)'.format(cls_pred_name[i], round(cls_pred_perc[i])))

    def evaluate(self, path, json_file):
        """
        Calculate the f1-score of the model on a dataset of images. The images must be
        in single folder.

        Arguments:
            path (str): location of the dataset
            json_file (str): path to json file containing image ids and their associated labels
        """

        generator = self._load_dataset(path, json_file)
        results = self.model.evaluate(generator)
        print('f1-score of', round(results[-1] * 100, 3), '%')

    def generate_metrics(self, path, json_file, threshold=0.5):
        """
        Computes classification report and confusion matrix resulting from predictions on
        a dataset of images and prints the results. Images must be located in a single folder.

        Arguments:
            path (str): location of the images
            json_file (str): path to json file containing image ids and their associated labels
            threshold (int): threshold for prediction (default to 0.5)
        """

        # getting list of true and predicted labels
        generator = self._load_dataset(path, json_file)
        cls_pred = self.model.predict(generator)
        cls_pred = cls_pred > threshold
        cls_true = np.array([generator.next()[1][0] for i in range(generator.n)])
        print('Labels & Predictions loaded for reports')

        # classification report
        print("\nClassification report")
        print(classification_report(cls_true, cls_pred, target_names=self.categories, digits=4))

        # confusion matrix
        print("\n Confusion matrix")
        for i, item in enumerate(multilabel_confusion_matrix(cls_true, cls_pred)):
            print(self.categories[i], '\n', item, '\n')
        
    def create_movie(self, classify_images, path="", threshold=0.5, json_file="", image_folder=""):
        """
        Create a movie from classified images.

        Arguments:
            path (str): location of the images
            classify_images (bool): location of the classified images, set this to False if you have the images already saved
            threshold (int): threshold for prediction (default to 0.5)
            json_file (str): path to json file containing image ids and their associated labels
            image_folder (str): if using already classified saved images, provide path to those images. 
        """

        if classify_images:
            self.classify_images(path, json_file, threshold, plot=True, save_img=True)
            image_folder = "classified_images/"
        
        video_name = 'video.avi'

        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 1, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()
