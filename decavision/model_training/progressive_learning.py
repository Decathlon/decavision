import numpy as np
import tensorflow as tf

from decavision.model_training.tfrecords_image_classifier import ImageClassifier
from decavision.utils import training_utils
from decavision.utils import utils


AUTO = tf.data.experimental.AUTOTUNE


class ProgressiveLearner(ImageClassifier):
    """
    Class to update an already trained model with new classes without losing too
    much of the information learned about the old classes.

    Arguments:
        tfrecords_folder (str): location of tfrecords (can be on google storage if authenticated), saved in
            folders train and val, filenames of the form filenumber-numberofimages.tfrec
        model_path (str): path to .h5 model trained with this library on the old classes
        transfer_model (str): pretrained model that was used to train the old model, can be one of Inception,
            Xception, Inception_Resnet, Resnet, B0, B3, B5, B7, V2-S, V2-M, V2-L or V2-XL
        batch_size (int): size of batches of data used for training
    """

    def __init__(self, tfrecords_folder, model_path, transfer_model, batch_size=128):
        super().__init__(tfrecords_folder, batch_size, transfer_model)
        self.distil = False
        self.temp = 5
        self.L = 5
        self.old_model = utils.load_model(model_path)
        self.logits = self.old_model.layers[-2].output

    def _update_model(self):
        """
        Take the old model, remove the classification layer and add a new one that includes
        the new classes. The weights connecting to the old classes are kept.

        Returns:
            tf.keras model: model with updated classification layer
        """
        self.old_classes = self.old_model.layers[-1].output_shape[-1]
        self.updated_classes = len(self.categories)
        self.new_classes = self.updated_classes - self.old_classes
        print('Updating output layer from {} classes to {} classes'.format(self.old_classes, self.updated_classes))

        # get weights connecting last layer to output and initialize new weights using them
        weights = self.old_model.get_weights()[-2]
        weights = np.append(weights, np.random.uniform(-1, 1, size=(weights.shape[0], self.new_classes)), axis=1)
        bias = self.old_model.get_weights()[-1]
        bias = np.append(bias, np.zeros((self.new_classes,)))

        # define new model without the old ouputs and add new output
        updated_logits = tf.keras.layers.Dense(self.updated_classes, weights=(weights, bias), name='updated_logs')(
            self.old_model.layers[-3].output)
        updated_predictions = tf.keras.layers.Activation('softmax', name='updated_preds')(updated_logits)

        return tf.keras.models.Model(inputs=self.old_model.inputs, outputs=updated_predictions)

    def _create_model(self, *args, **kwargs):
        """
        Create a keras model from a model with fewer classes.

        Returns:
            tf.keras model: pretrained model with added classes
            int: index of the layer where the last block of the pretrained model starts
            str: name of the loss function
            list(str): names of the metrics to use
        """
        print('Creating model')
        # get the last block for the fine tuning process
        if self.transfer_model == 'Xception':
            base_model_last_block = 116  # last block 126, two blocks 116
        elif self.transfer_model == 'Inception_Resnet':
            base_model_last_block = 287  # last block 630, two blocks 287
        elif self.transfer_model == 'Resnet':
            base_model_last_block = 155  # last block 165, two blocks 155
        elif self.transfer_model == 'B0':
            base_model_last_block = 213  # last block 229, two blocks 213
        elif self.transfer_model == 'B3':
            base_model_last_block = 354  # last block 370, two blocks 354
        elif self.transfer_model == 'B5':
            base_model_last_block = 417  # last block 559, two blocks 417
        elif self.transfer_model == 'B7':
            base_model_last_block = None  # all layers trainable
        elif self.transfer_model in ['V2-S', 'V2-M', 'V2-L', 'V2-XL']:
            base_model_last_block = None  # all layers trainable
        else:
            base_model_last_block = 249  # last block 280, two blocks 249

        self.model = self._update_model()
        self.updated_logits = self.model.layers[-2].output

        # Define the optimizer and the loss and the optimizer
        loss = 'sparse_categorical_crossentropy'
        if self.distil and not self.use_TPU:
            loss = training_utils.custom_loss(self.logits, self.updated_logits, self.old_classes, self.temp, self.L)
        metrics = ['sparse_categorical_accuracy']

        return self.model, base_model_last_block, loss, metrics

    def fit(self, learning_rate=1e-3, learning_rate_fine_tuning=1e-4,
            epochs=5, save_model=False, verbose=True,
            fine_tuning=True, logs=None):
        """
        Train an image classification model based on a model trained with a smaller number of classes.
        The whole model is trained, unless there is some fine tuning, in which case a second round of
        training is done with the last layers of the model unfrozen. Training can be stopped if
        no sufficient improvement in accuracy.

        Arguments:
            learning_rate (float): learning rate used when training whole model
            learning_rate_fine_tuning (float): learning rate used when fine tuning last layers
            epochs (int): number of epochs done when training (doubled if fine tuning)
            save_model (str): specify a name for the trained model to save it, model is saved in .h5 if
                the name contains the extension and in .pb if no extension in the name
            verbose (bool): show details of training or not
            fine_tuning (bool): fine tune last layers or not
            min_accuracy (float): if specified, stop training when improvement in accuracy is smaller than min_accuracy
            logs (str): if specified, tensorboard is used and logs are saved at this location
        """
        super().fit(learning_rate=learning_rate, learning_rate_fine_tuning=learning_rate_fine_tuning,
                    epochs=epochs, save_model=save_model, verbose=verbose,
                    fine_tuning=fine_tuning, logs=logs)

    def hyperparameter_optimization(self):
        """
        This class is not implemented for progressive learning.
        """
        pass
