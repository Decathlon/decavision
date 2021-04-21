import os
import pandas as pd
import tensorflow as tf
import numpy as np
import shutil
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from distutils.dir_util import copy_tree
from tqdm import tqdm


class PseudoLabelGenerator:
    """
    Class to make pseudo labels.
    """

    def __init__(self):
        pass

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

    def generate_pseudolabel_data(self, model_path,
                                  train_data_path, unlabeled_path,
                                  pseudo_data_path, save_confidences_csv=False,
                                  plot_confidences=False, threshold=None):
        """ Convert image and label to tfrecord example.
            Arguments:
                model_path (str): path to trained model
                train_path (str): path to training data
                unlabeled_path (str): path to unlabeled data
                pseudo_path (str): path to store train data and pseudo data
                save_confidences_csv (boolean): Whether to save CSV file with pseudo confidences, default is False.
                plot_confidences (boolean): Whether to plot confidence graphs for raw confidences and per class confidences.
                threshold (float): Discard images with prediction below this confidence, default is None.

            Returns:
                pseudo_data_path: A folder with both labeled and pseudo labeled images.
        """
        # Copy the training/labeled data to the destination folder where
        # we will also store the pseudo labels.
        copy_tree(train_data_path, pseudo_data_path)

        # Load model
        model = load_model(model_path, compile=False)
        print("Loaded model.")

        # Make dictionary for classes and their index
        class_names = sorted(os.listdir(train_data_path))
        class_dict = {}
        index = 0
        for cat in class_names:
            class_dict[cat] = index
            index += 1

        print("Generating pseudo labels...")
        # Generate pseudo labels
        unlabeled_image_paths = os.listdir(unlabeled_path)
        print("There are {} unlabeled images.".format(
            len(unlabeled_image_paths)))
        raw_predictions_paths = []
        raw_predictions = []  # single confidence value of predicted class
        raw_predictions_all = []  # confidences for all classes
        for path in tqdm(unlabeled_image_paths[:100]):
            # Load the image
            img = np.array(self._load_img(os.path.join(unlabeled_path, path),
                                          target_size=(299, 299)))
            # Make predictions using trained model
            y_pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
            # Get class index
            y = np.argmax(y_pred)
            # Get probability score
            proba = y_pred[0][y]

            raw_predictions.append(proba)
            raw_predictions_all.append(y_pred)
            raw_predictions_paths.append(os.path.join(unlabeled_path, path))

            # The results variable should be the same as the class category
            for class_name, index in class_dict.items():
                if threshold:
                    # For thresholding predictions
                    if index == y and proba >= threshold:
                        shutil.copy(os.path.join(unlabeled_path, path),
                                    os.path.join(pseudo_data_path, class_name))
                else:
                    # For hard predictions
                    if index == y:
                        shutil.copy(os.path.join(unlabeled_path, path),
                                    os.path.join(pseudo_data_path, class_name))
        print("Generated pseudo labels.")

        if not os.path.exists("outputs"):
            os.mkdir("outputs")
        if save_confidences_csv:
            print("Saving CSV with pseudo predictions.")
            data = {'filepaths': raw_predictions_paths,
                    'class preds highest': raw_predictions,
                    'class preds all': raw_predictions_all}
            df = pd.DataFrame(data)
            df.to_csv("outputs/pseudo_predictions.csv", index=False)

        if plot_confidences:
            # Plot graph for highest confidence pseudo labels for each class
            for idx in range(len(class_names)):
                predictions = []
                samples = []
                for i in raw_predictions_all:
                    predictions.append(i[0][idx])

                predictions = sorted(predictions)
                samples = [pred for pred in range(len(predictions))]
                plt.bar(samples, predictions, color='g')
                plt.axhline(y=0.5, color='r', linestyle='--')
                plt.title("Confidences for the class: {}".format(
                    class_names[idx]), size=16)
                plt.xlabel("Number of unlabelled images", size=16)
                plt.ylim([0.0, 1.0])
                plt.ylabel("Probability", size=16)
                plt.tick_params(labelright=True)
                plt.savefig(
                    "outputs/{}_confidences.png".format(class_names[idx]), dpi=100)
                plt.clf()  # clear buffer, otherwise plot overlap!

            # Plot graph for highest confidence pseudo labels for all unlabeled images
            predictions_highest = sorted(raw_predictions)
            samples_highest = [pred for pred in range(len(raw_predictions))]
            plt.bar(samples_highest, predictions_highest, color='g')
            plt.title("Highest confidence pseudo labels", size=16)
            plt.xlabel("Number of unlabelled images", size=16)
            plt.ylim([0.0, 1.0])
            plt.ylabel("Probability", size=16)
            plt.tick_params(labelright=True)
            plt.savefig("outputs/highest_confidence_predictions.png", dpi=100)
            plt.clf()


if __name__ == '__main__':
    transformer = PseudoLabelGenerator()
