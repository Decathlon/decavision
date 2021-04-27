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
        model_path (str): location of the h5 tensorflow model to use
        train_data_path (str): folder which holds training data
        unlabeled_path (str): folder which holds unlabeled data
        pseudo_data_path (str): folder to store training data and pseudo data combined
        output_folder (str): folder to store outputs
        csv_filename (str): name of csv file
    """

    def __init__(self, model_path="model.h5", train_data_path="data/image_dataset/train",
                 unlabeled_path="data/unlabeled", pseudo_data_path="data/train_ssl",
                 output_folder="outputs", csv_filename="data.csv"):

        self.train_data_path = train_data_path
        self.unlabeled_path = unlabeled_path
        self.pseudo_data_path = pseudo_data_path
        self.output_folder = output_folder
        self.csv_path = os.path.join(self.output_folder, csv_filename)

        # Load model
        self.model = load_model(model_path, compile=False)
        print("Loaded model.")

        # Make new output folder
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

        # Make dictionary for classes and their index
        self.class_names = sorted(os.listdir(self.train_data_path))
        self.class_dict = {cat: i for (i, cat) in enumerate(self.class_names)}

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
        Plots a bar plot and saves it to a file.

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

    def plot_confidence_scores(self, per_class=True, overall=True):
        """
        Generate bar plots for highest confidence predictions per class and overall and save them.

        Arguments:
            per_class (bool): make bar plots per class or not
            overall (bool): make overall bar plot or not
        """
        dt = pd.read_csv(self.csv_path)
        dt['All Class Predictions List'] = dt['All Class Predictions'].apply(
            lambda x: ast.literal_eval(x))

        raw_predictions_ = dt[["Highest Confidence"]].values
        raw_predictions = [pred[0] for pred in raw_predictions_]

        raw_predictions_all_ = dt[['All Class Predictions List']].values
        raw_predictions_all = [pred[0] for pred in raw_predictions_all_]

        # Plot graph for highest confidence pseudo labels for each class
        if per_class:
            for idx, cat in enumerate(self.class_names):
                predictions = [pred[idx] for pred in raw_predictions_all]
                title = "Confidences for the class: {}".format(cat)
                path = "{}/{}_confidences.png".format(
                    self.output_folder, cat)
                self._plot_data(predictions, title, path)

        # Plot graph for highest confidence pseudo labels for all unlabeled images
        if overall:
            self._plot_data(raw_predictions,
                            name="Highest confidence pseudo labels",
                            output_path="{}/highest_confidence_predictions.png".format(
                                self.output_folder))

    def make_dataset(self, filenames, batch_size):
        def parse_image(filename):
            image = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [299, 299])
            image = tf.cast(image, tf.uint8)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            return image

        def configure_for_performance(ds):
            ds = ds.batch(batch_size)
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return ds

        filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
        images_ds = filenames_ds.map(parse_image,
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = configure_for_performance(images_ds)
        return ds

    def move_unlabeled_images(self, threshold):
        """
        Split unlabeled images into folders based on pseudo labels. A copy of the unsplit images is kept.

        Arguments:
            threshold (float): Discard images with prediction below this confidence, default is None.
        """

        # Copy the training/labeled data to the destination folder where
        # we will also store the pseudo labels.
        copy_tree(self.train_data_path, self.pseudo_data_path)

        dt = pd.read_csv(self.csv_path)
        filepaths = dt[["Filepaths"]].values
        predicted_class = dt[["Predicted Class"]].values
        raw_predictions = dt[["Highest Confidence"]].values

        for proba, y, path in zip(raw_predictions, predicted_class, filepaths):
            # The results variable should be the same as the class category
            for class_name, index in self.class_dict.items():
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
        print("Moved unlabeled images to their pseudo label categories.")

    def generate_pseudolabel_data(self, plot_confidences=False, threshold=None, move_images=False, batch_size=32):
        """ Use trained model to make pseudo labels and save them into a csv file. Also possible to plot the results
            and move the unlabeled images directly to the category corresponding to their pseudo label.

            Arguments:
                plot_confidences (boolean): Whether to plot confidence graphs for raw confidences and per class confidences.
                threshold (float): Discard images with prediction below this confidence, default is None.
                move_images (bool): Move images into categories or not
                batch_size (int): Batch size while making predictions

            Returns:
                pseudo_data_path: A folder with both labeled and pseudo labeled images.
        """

        print("Generating pseudo labels...")
        # Generate pseudo labels
        unlabeled_image_paths = os.listdir(self.unlabeled_path)
        print("There are {} unlabeled images.".format(
            len(unlabeled_image_paths)))

        raw_predictions_paths = []
        raw_predictions = []  # single confidence value of predicted class
        predicted_class = []  # predicted class index
        raw_predictions_all = []  # confidences for all classes

        unlabeled_filenames = [os.path.join(self.unlabeled_path,
                                            path) for path in unlabeled_image_paths]
        ds = self.make_dataset(unlabeled_filenames, batch_size)
        y_preds = self.model.predict(ds)
        #import pdb; pdb.set_trace()
        for y_pred in y_preds:
            y = np.argmax(y_pred)
            # Get probability score
            proba = y_pred[y]
            predicted_class.append(y)
            raw_predictions.append(proba)
            raw_predictions_all.append(list(y_pred))
        raw_predictions_paths = [path for path in unlabeled_image_paths]

        # 'Pseudo Class Names': pseudo_class_names,
        print("Saving CSV with pseudo predictions.")
        data = {'Filepaths': raw_predictions_paths,
                'Predicted Class': predicted_class,
                'Highest Confidence': raw_predictions,
                'All Class Predictions': raw_predictions_all}
        df = pd.DataFrame(data)
        df.to_csv(self.csv_path, index=False)

        # move pseudo labeled images
        if move_images:
            self.move_unlabeled_images(threshold=threshold)

        if plot_confidences:
            print("Plotting data.")
            self.plot_confidence_scores()
