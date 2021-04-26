import ast
import os
import shutil
from distutils.dir_util import copy_tree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm


class PseudoLabelGenerator:
    """
    Class to generate pseudo labels.
    Arguments:
        model_path (str): folder which stores the trained model
        train_path (str): folder which holds training data
        unlabeled_path (str): folder which holds unlabeled data
        pseudo_path (str): folder to store training data and pseudo data combined
        output_folder (str): folder to store outputs
        csv_path (str): name of csv file
    """

    def __init__(self, model_path="model", train_data_path="data/image_dataset/train", unlabeled_path="data/image_dataset/unlabeled", pseudo_data_path="data/image_dataset/train_ssl", output_folder="outputs", csv_path="data.csv"):
        self.model_path = model_path
        self.train_data_path = train_data_path
        self.unlabeled_path = unlabeled_path
        self.pseudo_data_path = pseudo_data_path
        self.output_folder = output_folder
        self.csv_path = csv_path

        # Load model
        self.model = load_model(self.model_path, compile=False)
        print("Loaded model.")

        # Make new output folder
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

    def _load_img(self, path, target_size=(299, 299)):
        """
        Load an image from a given path and normalize it

        Args:
            path (list): Input image path
            target_size (tuple): Size of image
        Returns:
            np.array: Numpy array of the data
        """
        # Read image
        bits = tf.io.read_file(path)
        image = tf.image.decode_jpeg(bits, channels=3)
        # Resize
        image = tf.image.resize(image, size=[*target_size])
        image = tf.reshape(image, [*target_size, 3])
        image = tf.cast(image, tf.uint8)
        # Normalize [0, 1]
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = image.numpy()
        return image

    def _plot_data(self, predictions, name, output_path):
        """
        Plots a bar plot.

        Arguments:
            predictions (list): List of predictions
            name (str): Title of the plot
            output_path (str): Save file using this name
        """
        predictions = sorted(predictions)
        samples = list(range(len(predictions)))
        plt.bar(samples, predictions, color='g')
        plt.axhline(y=0.5, color='r', linestyle='--')
        plt.title(name, size=16)
        plt.xlabel("Number of unlabelled images", size=16)
        plt.ylim([0.0, 1.0])
        plt.ylabel("Probability", size=16)
        plt.tick_params(labelright=True)
        plt.savefig(output_path, dpi=100)
        plt.clf()  # clear buffer, otherwise plot overlap!

    def _plot_confidence_scores(self, classes):
        """
        Save pseudo label names and their confidence scores in a csv file.

        Arguments:
            classes (list): List containing the class names
        """
        dt = pd.read_csv(os.path.join(self.output_folder, self.csv_path))
        dt['All Class Predictions List'] = dt['All Class Predictions'].apply(
            lambda x: ast.literal_eval(x))

        raw_predictions_ = dt[["Highest Confidence"]].values
        raw_predictions = [pred[0] for pred in raw_predictions_]

        raw_predictions_all_ = dt[['All Class Predictions List']].values
        raw_predictions_all = [pred[0] for pred in raw_predictions_all_]

        # Plot graph for highest confidence pseudo labels for each class

        for idx, _ in enumerate(classes):
            predictions = [pred[idx] for pred in raw_predictions_all]
            title = "Confidences for the class: {}".format(classes[idx])
            path = "{}/{}_confidences.png".format(
                self.output_folder, classes[idx])
            self._plot_data(predictions, title, path)

        # Plot graph for highest confidence pseudo labels for all unlabeled images
        self._plot_data(raw_predictions,
                        name="Highest confidence pseudo labels",
                        output_path="{}/highest_confidence_predictions.png".format(
                            self.output_folder))

    def _move_unlabeled_images(self, threshold, dictionary):
        """
        Save pseudo labels.

        Arguments:
            threshold (float): Discard images with prediction below this confidence, default is None.
            dictionary (dict): Class dictionary
        """

        # Copy the training/labeled data to the destination folder where
        # we will also store the pseudo labels.
        copy_tree(self.train_data_path, self.pseudo_data_path)

        dt = pd.read_csv(os.path.join(self.output_folder, self.csv_path))
        filepaths = dt[["Filepaths"]].values
        predicted_class = dt[["Predicted Class"]].values
        raw_predictions = dt[["Highest Confidence"]].values

        for proba, y, path in zip(raw_predictions, predicted_class, filepaths):
            # The results variable should be the same as the class category
            for class_name, index in dictionary.items():
                if threshold:
                    # For thresholding predictions
                    if index == y and proba >= threshold:
                        shutil.copy(os.path.join(self.unlabeled_path, str(path[0])),
                                    os.path.join(self.pseudo_data_path, class_name))
                else:
                    # For hard predictions
                    if index == y:
                        shutil.copy(os.path.join(self.unlabeled_path, str(path[0])),
                                    os.path.join(self.pseudo_data_path, class_name))
        print("Saved pseudo labels.")

    def generate_pseudolabel_data(self, save_confidences_csv=False,
                                  plot_confidences=False, threshold=None):
        """ Convert image and label to tfrecord example.
            Arguments:
                save_confidences_csv (boolean): Whether to save CSV file with pseudo confidences, default is False.
                plot_confidences (boolean): Whether to plot confidence graphs for raw confidences and per class confidences.
                threshold (float): Discard images with prediction below this confidence, default is None.

            Returns:
                pseudo_data_path: A folder with both labeled and pseudo labeled images.
        """

        # Make dictionary for classes and their index
        class_names = sorted(os.listdir(self.train_data_path))
        class_dict = {cat: i for (i, cat) in enumerate(class_names)}

        print("Generating pseudo labels...")
        # Generate pseudo labels
        unlabeled_image_paths = os.listdir(self.unlabeled_path)
        print("There are {} unlabeled images.".format(
            len(unlabeled_image_paths)))
        raw_predictions_paths = []
        raw_predictions = []  # single confidence value of predicted class
        predicted_class = []  # predicted class index
        raw_predictions_all = []  # confidences for all classes
        for path in tqdm(unlabeled_image_paths):
            # Load the image
            img = np.array(self._load_img(os.path.join(self.unlabeled_path, path),
                                          target_size=(299, 299)))
            # Make predictions using trained model
            y_pred = self.model.predict(np.expand_dims(img, axis=0), verbose=0)
            # Get class index
            y = np.argmax(y_pred)
            # Get probability score
            proba = y_pred[0][y]

            predicted_class.append(y)
            raw_predictions.append(proba)
            raw_predictions_all.append(list(y_pred[0]))
            raw_predictions_paths.append(path)

        if not os.path.exists("outputs"):
            os.mkdir("outputs")
        if save_confidences_csv:
            # 'Pseudo Class Names': pseudo_class_names,
            print("Saving CSV with pseudo predictions.")
            data = {'Filepaths': raw_predictions_paths,
                    'Predicted Class': predicted_class,
                    'Highest Confidence': raw_predictions,
                    'All Class Predictions': raw_predictions_all}
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(self.output_folder,
                      self.csv_path), index=False)

        # Save pseudo labeled images
        self._move_unlabeled_images(threshold=threshold,
                                    dictionary=class_dict)

        if plot_confidences:
            # Plot only if there are any CSV files.
            if save_confidences_csv:
                print("Plotting data.")
                self._plot_confidence_scores(classes=class_names)
            else:
                print("No CSV file is present to plot the data.")
