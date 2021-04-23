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
        model_path (str): path to trained model
        train_path (str): path to training data
        unlabeled_path (str): path to unlabeled data
        pseudo_path (str): path to store train data and pseudo data
    """

    def __init__(self, model_path, train_data_path, unlabeled_path, pseudo_data_path):
        self.model_path = model_path
        self.train_data_path = train_data_path
        self.unlabeled_path = unlabeled_path
        self.pseudo_data_path = pseudo_data_path

        # Load model
        model = load_model(self.model_path, compile=False)
        self.model = model
        print("Loaded model.")

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
        samples = [pred for pred in range(len(predictions))]
        plt.bar(samples, predictions, color='g')
        plt.axhline(y=0.5, color='r', linestyle='--')
        plt.title(name, size=16)
        plt.xlabel("Number of unlabelled images", size=16)
        plt.ylim([0.0, 1.0])
        plt.ylabel("Probability", size=16)
        plt.tick_params(labelright=True)
        plt.savefig(output_path, dpi=100)
        plt.clf()  # clear buffer, otherwise plot overlap!

    def _plot_confidence_scores(self, csv_path, classes):
        """
        Save pseudo label names and their confidence scores in a csv file.

        Arguments:
            csv_path (str): Path to save SCV file
            classes (list): List containing the class names
        """
        dt = pd.read_csv(csv_path)
        dt['All Class Predictions List'] = dt['All Class Predictions'].apply(
            lambda x: ast.literal_eval(x))

        raw_predictions_ = dt[["Highest Confidence"]].values
        raw_predictions = [raw_predictions_[idx][0]
                           for idx in range(len(raw_predictions_))]

        raw_predictions_all_ = dt[['All Class Predictions List']].values
        raw_predictions_all = [raw_predictions_all_[idx][0]
                               for idx in range(len(raw_predictions_all_))]

        # Plot graph for highest confidence pseudo labels for each class
        for idx in range(len(classes)):
            predictions = []
            for i in raw_predictions_all:
                predictions.append(i[idx])

            title = "Confidences for the class: {}".format(classes[idx])
            path = "outputs/{}_confidences.png".format(classes[idx])
            self._plot_data(predictions, title, path)

        # Plot graph for highest confidence pseudo labels for all unlabeled images
        self._plot_data(raw_predictions,
                        name="Highest confidence pseudo labels",
                        output_path="outputs/highest_confidence_predictions.png")

    def _save_pseudo_labels(self, csv_path, threshold, dictionary):
        """
        Save pseudo labels.

        Arguments:
            csv_path (str): Path to CSV file
            threshold (float): Discard images with prediction below this confidence, default is None.
            dictionary (dict): Class dictionary
        """

        dt = pd.read_csv(csv_path)
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

        # Copy the training/labeled data to the destination folder where
        # we will also store the pseudo labels.
        copy_tree(self.train_data_path, self.pseudo_data_path)

        # Make dictionary for classes and their index
        class_names = sorted(os.listdir(self.train_data_path))
        class_dict = {}
        index = 0
        for cat in class_names:
            class_dict[cat] = index
            index += 1

        print("Generating pseudo labels...")
        # Generate pseudo labels
        unlabeled_image_paths = os.listdir(self.unlabeled_path)
        print("There are {} unlabeled images.".format(
            len(unlabeled_image_paths)))
        raw_predictions_paths = []
        raw_predictions = []  # single confidence value of predicted class
        predicted_class = []  # predicted class index
        raw_predictions_all = []  # confidences for all classes
        for path in tqdm(unlabeled_image_paths[:100]):
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
            df.to_csv("outputs/data.csv", index=False)

        # Save pseudo labeled images
        self._save_pseudo_labels(csv_path="outputs/data.csv",
                                 threshold=threshold,
                                 dictionary=class_dict)

        if plot_confidences:
            # Plot only if there are any CSV files.
            if save_confidences_csv:
                print("Plotting data.")
                self._plot_confidence_scores(
                    csv_path="outputs/data.csv", classes=class_names)
            else:
                print("No CSV file is present to plot the data.")


if __name__ == '__main__':
    transformer = PseudoLabelGenerator()
