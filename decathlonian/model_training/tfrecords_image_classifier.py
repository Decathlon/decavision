import datetime
import logging
import math
import os

import dill
from efficientnet.tfkeras import EfficientNetB0, EfficientNetB3, EfficientNetB5
import skopt
import tensorflow as tf

from decathlonian.utils import training_utils
from decathlonian.utils import utils

AUTO = tf.data.experimental.AUTOTUNE


class ImageClassifier:
    """
    Class to train an image classification model by using transfer learning.
    A hyperparameter optimization tool is also provided. Data must be saved in tfrecords format.
    See data_preparation.generate_tfrecords to go from image data to tfrecords.

    Arguments:
        tfrecords_folder (str): location of tfrecords (can be on google storage if authenticated), saved in
            folders train and val, filenames of the form filenumber-numberofimages.tfrec
        batch_size (int): size of batches of data used for training
        transfer_model (str): pretrained model to use for transfer learning, can be one of Inception,
            Xception, Inception_Resnet, Resnet, B0, B3 or B5
    """

    def __init__(self, tfrecords_folder, batch_size=128, transfer_model='Inception'):

        self.tfrecords_folder = tfrecords_folder
        self.use_TPU, self.use_GPU = utils.check_PU()
        if self.use_TPU and batch_size % 8:
            print('Batch size {} is not multiple of 8, required for TPU'.format(batch_size))
            batch_size = 8 * round(batch_size/8)
            print('New batch size is {}'.format(batch_size))
        self.batch_size = batch_size
        self.transfer_model = transfer_model
        
        # We expect the classes to be saved in the same folder where tfrecords are
        self.categories = utils.load_classes(tfrecords_folder)
        print('Classes ({}) :'.format(len(self.categories)))
        print(self.categories)

        train_tfrecords = tf.io.gfile.listdir(
            os.path.join(tfrecords_folder, 'train'))
        self.nb_train_shards = len(train_tfrecords)
        print('Training tfrecords = {}'.format(self.nb_train_shards))
        
        val_tfrecords = tf.io.gfile.listdir(
            os.path.join(tfrecords_folder, 'val'))
        self.nb_val_shards = len(val_tfrecords)
        print('Val tfrecords = {}'.format(self.nb_val_shards))
        
        # Expected tfrecord file name : filenumber-numberofimages.tfrec (02-2223.tfrec)
        self.nb_train_images = 0
        for train_tfrecord in train_tfrecords:
            self.nb_train_images += int(train_tfrecord.split('.')[0].split('-')[1])
        print('Training images = '+str(self.nb_train_images))
        
        nb_val_images = 0
        for val_tfrecord in val_tfrecords:
            nb_val_images += int(val_tfrecord.split('.')[0].split('-')[1])
        print('Val images = '+str(nb_val_images))
        
        self.training_shard_size = math.ceil(self.nb_train_images/self.nb_train_shards)
        print('Training shard size = {}'.format(self.training_shard_size))

        val_shard_size = math.ceil(nb_val_images/self.nb_val_shards)
        print('Val shard size = {}'.format(val_shard_size))

        print('Training batch size = '+str(self.batch_size))
        self.steps_per_epoch = int(self.nb_train_images / self.batch_size)
        print('Training steps per epochs = '+str(self.steps_per_epoch))

        print('Val batch size = '+str(self.batch_size))
        self.validation_steps = int(nb_val_images / self.batch_size)
        print('Val steps per epochs = '+str(self.validation_steps))

        if transfer_model in ['Inception', 'Xception', 'Inception_Resnet', 'B3', 'B5']:
            self.target_size = (299, 299)
        else:
            self.target_size = (224, 224)

    def _get_dataset(self, is_training, nb_readers):
        """
        Extract data from tfrecords into a tf.data iterator. If data is used for training, it is shuffled each time
        the iterator is used. Strongly inspired by
        https://colab.research.google.com/github/GoogleCloudPlatform/training-data-analyst/blob/master/courses/fast-and-lean-data-science/07_Keras_Flowers_TPU_playground.ipynb#scrollTo=LtAVr-4CP1rp

        Arguments:
            is_training (bool): set if data used for training or not, determines if data is taken from train (True)
                or val (False) tfrecords
            nb_readers (int): number of different files to combine

        Returns:
            tf.data.dataset: iterable dataset with content of relevant tfrecords (images and labels)
        """
        
        def _read_tfrecord(example):
            """ Extract image and label from single tfrecords example."""
            features = {
                    'image': tf.io.FixedLenFeature((), tf.string),
                    'label': tf.io.FixedLenFeature((), tf.int64),
                }
            example = tf.io.parse_single_example(example, features)
                
            image = tf.image.decode_jpeg(example['image'], channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            feature = tf.image.resize(image, [*self.target_size])
            label = tf.cast([example['label']], tf.int32)
            return feature, label

        def _load_dataset(filenames):
            """ Load the tfrecords files in a tf.data object."""
            buffer_size = 8 * 1024 * 1024  # 8 MiB per file
            dataset = tf.data.TFRecordDataset(
                filenames, buffer_size=buffer_size)
            return dataset
    
        file_pattern = os.path.join(self.tfrecords_folder, "train/*" if is_training else "val/*")
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)

        # Enable non-determinism only for training.
        options = tf.data.Options()
        options.experimental_deterministic = not is_training
        dataset = dataset.with_options(options)
        dataset = dataset.interleave(_load_dataset, nb_readers, num_parallel_calls=AUTO)
        if is_training:
            # Shuffle only for training.
            dataset = dataset.shuffle(buffer_size=math.ceil(self.training_shard_size*self.nb_train_shards/4))
        dataset = dataset.repeat()
        dataset = dataset.map(_read_tfrecord, num_parallel_calls=AUTO)
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=self.use_TPU)
        dataset = dataset.prefetch(AUTO)
        return dataset

    def get_training_dataset(self):
        """
        Extract data from training tfrecords located in tfrecords_folder. Data is shuffled.

        Returns:
            tf.data.dataset: iterable dataset with content of training tfrecords (images and labels)
        """
        return self._get_dataset(True, self.nb_train_shards)

    def get_validation_dataset(self):
        """ Extract data from validation tfrecords located in tfrecords_folder.

        Returns:
            tf.data.dataset: iterable dataset with content of validation tfrecords (images and labels)
        """
        return self._get_dataset(False, self.nb_val_shards)

    def _create_model(self, activation, hidden_size, dropout, l2_lambda):
        """
        Create a keras model from a pretrained model. Add a classification layer on top and
        possibly an extra Dense layer (with dropout and batchnorm)

        Arguments:
            activation (str): activation function to use in extra layer, any keras activation is possible
            hidden_size (int): number of neurons in extra layer, no layer if 0
            dropout (float): rate of dropout to use in extra layer (<1)
            l2_lambda (float): amount of L2 regularization to include in extra layer

        Returns:
            tf.keras model: pretrained model with added layers
            int: index of the layer where the last block of the pretrained model starts
            str: name of the loss function
            list(str): names of the metrics to use
        """
        print('Creating model')
        # load the pretrained model, without the classification (top) layers
        if self.transfer_model == 'Xception':
            base_model = tf.keras.applications.Xception(weights='imagenet',
                                                        include_top=False, input_shape=(*self.target_size, 3))
            base_model_last_block = 116  # last block 126, two blocks 116
        elif self.transfer_model == 'Inception_Resnet':
            base_model = tf.keras.applications.InceptionResNetV2(
                weights='imagenet', include_top=False, input_shape=(*self.target_size, 3))
            base_model_last_block = 287  # last block 630, two blocks 287
        elif self.transfer_model == 'Resnet':
            base_model = tf.keras.applications.ResNet50(weights='imagenet',
                                                        include_top=False, input_shape=(*self.target_size, 3))
            base_model_last_block = 155  # last block 165, two blocks 155
        elif self.transfer_model == 'B0':
            base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                        input_shape=(*self.target_size, 3))
            base_model_last_block = 213  # last block 229, two blocks 213
        elif self.transfer_model == 'B3':
            base_model = EfficientNetB3(weights='imagenet', include_top=False,
                                        input_shape=(*self.target_size, 3))
            base_model_last_block = 354  # last block 370, two blocks 354
        elif self.transfer_model == 'B5':
            base_model = EfficientNetB5(weights='imagenet', include_top=False,
                                        input_shape=(*self.target_size, 3))
            base_model_last_block = 417  # last block 559, two blocks 417
        else:
            base_model = tf.keras.applications.InceptionV3(weights='imagenet',
                                                           include_top=False, input_shape=(*self.target_size, 3))
            base_model_last_block = 249  # last block 280, two blocks 249

        # Set only the top layers as trainable (if we want to do fine-tuning,
        # we can train the base layers as a second step)
        base_model.trainable = False

        # Add the classification layers using Keras functional API
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # Hidden layer for classification
        if hidden_size == 0:
            x = tf.keras.layers.Dropout(rate=dropout)(x)
        else:
            x = tf.keras.layers.Dense(hidden_size, use_bias=False,
                                      kernel_regularizer=tf.keras.regularizers.l2(l=l2_lambda))(x)
            # scale: When the next layer is linear (also e.g. nn.relu), this can be disabled since the
            # scaling can be done by the next layer.
            x = tf.keras.layers.BatchNormalization(scale=activation != 'relu')(x)
            x = tf.keras.layers.Activation(activation=activation)(x)
            x = tf.keras.layers.Dropout(rate=dropout)(x)

        x = tf.keras.layers.Dense(len(self.categories), name='logs')(x)  # Output layer
        predictions = tf.keras.layers.Activation('softmax', name='preds')(x)  # Output activation
        loss = 'sparse_categorical_crossentropy'
        metrics = ['sparse_categorical_accuracy']

        return tf.keras.Model(inputs=base_model.input, outputs=predictions), base_model_last_block, loss, metrics

    def fit(self, save_model=None, export_model=None, min_accuracy=None,
            epochs=5, hidden_size=1024, learning_rate=1e-3, learning_rate_fine_tuning=1e-4,
            dropout=0.5, l2_lambda=5e-4, fine_tuning=True,
            verbose=True, logs=None, activation='swish'):
        """
        Train an image classification model based on a pretrained model. A classification layer is added
        to the pretrained model, with potentially an extra combination of Dense, Dropout and Batchnorm.
        Only added layers are trained, unless there is some fine tuning, in which case a second round of
        training is done with the last block of the pretrained model unfrozen. Training can be stopped if
        no sufficient improvement in accuracy.

        Arguments:
            learning_rate (float): learning rate used when training extra layers
            learning_rate_fine_tuning (float): learning rate used when fine tuning pretrained model
            epochs (int): number of epochs done when training (doubled if fine tuning)
            activation (str): activation function to use in extra layer, any keras activation is possible
            hidden_size (int): number of neurons in extra layer, no layer if 0
            save_model (str): specify a name for the trained model to save it in .h5 format
            export_model (str): specify a name for the trained model to save it in .pb format
            dropout (float): rate of dropout to use in extra layer (<1)
            verbose (bool): show details of training or not
            fine_tuning (bool): fine tune pretrained model or not
            l2_lambda (float): amount of L2 regularization to include in extra layer
            min_accuracy (float): if specified, stop training when improvement in val accuracy is smaller than min_accuracy
            logs (str): if specified, tensorboard is used and logs are saved at this location
        """
        callbacks = None
        if logs:
            logdir = os.path.join(logs, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            print('Fit log dir : ' + logdir)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)
            callbacks = [tensorboard_callback]

        # if we want to stop training when no sufficient improvement in accuracy has been achieved
        if min_accuracy:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                                          min_delta=min_accuracy)
            if not callbacks:
                callbacks = [early_stop]
            else:
                callbacks.append(early_stop)
        
        # compile the model and fit the model
        if self.use_TPU:
            tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
            tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
            strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver)
            
            with strategy.scope():
                model, base_model_last_block, loss, metrics = self._create_model(activation, hidden_size, dropout, l2_lambda)
                print('Compiling for TPU')
                optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                    
        else:
            model, base_model_last_block, loss, metrics = self._create_model(activation, hidden_size, dropout, l2_lambda)
            print('Compiling for GPU') if self.use_GPU else print('Compiling for CPU')
            optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                
        print('Fitting')
        history = model.fit(self.get_training_dataset(), steps_per_epoch=self.steps_per_epoch, epochs=epochs,
                            validation_data=self.get_validation_dataset(), validation_steps=self.validation_steps,
                            verbose=verbose, callbacks=callbacks)

        # Fine-tune the model, if we wish so
        if fine_tuning and not model.stop_training:
            print('===========')
            print('Fine-tuning')
            print('===========')
            
            fine_tune_epochs = epochs
            total_epochs = epochs + fine_tune_epochs
            
            print('Unfreezing last block of layers from the base model')
            for layer in model.layers[:base_model_last_block]:
                layer.trainable = False
            # don't unfreeze batchnorm (see https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/)
            for layer in model.layers[base_model_last_block:]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True

            # Fit the model
            # we need to recompile the model for these modifications to take effect with a low learning rate
            print('Recompiling model')
            optimizer = tf.keras.optimizers.Adam(lr=learning_rate_fine_tuning)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                
            print('Fine tunning')
            history = model.fit(self.get_training_dataset(), steps_per_epoch=self.steps_per_epoch, epochs=total_epochs,
                                validation_data=self.get_validation_dataset(), validation_steps=self.validation_steps,
                                verbose=verbose, callbacks=callbacks, initial_epoch=epochs)

        # Evaluate the model, just to be sure
        self.fitness = history.history['val_sparse_categorical_accuracy'][-1]
        self.model = model
        del history
        del model
        # Save the model
        if save_model:
            self.model.save(save_model + '.h5')
            print('Model saved')
        if export_model:
            self.model.save(export_model)
            print('Model exported')

    def hyperparameter_optimization(self, num_iterations=20, n_random_starts=10, min_accuracy=None, save_results=False):
        """
        Try different combinations of hyperparameters to find the best model possible. Start by trying random
        combinations and after some time learn from th previous tries. Scikit-optimize checkoint is saved
        at each step in the working directory. If checkpoint present in working directory, optimization starts
        back from where it left off. Logs of all tries are also saved in working directory. Hyperparameters that
        are varied are epochs, hidden_size, learning_rate, learning_rate_fine_tuning, fine_tuning, dropout and
        l2_lambda.

        Arguments:
            n_random_starts (int): number of random combinations of hyperparameters first tried
            num_iterations (int): total number of hyperparameter combinations to try (aim for a 1:1 to 2:1 ratio
                num_iterations/n_random_starts)
            min_accuracy (float): if specified, training in single try will stop when improvement in accuracy
                is smaller than min_accuracy
            save_results (bool): decide to save optimal hyperparameters in pickle file when done
        """
        # initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("hyperparameters.log"),
                logging.StreamHandler()
            ]
        )
        # declare the hyperparameters search space
        dim_epochs = skopt.space.Integer(low=1, high=6, name='epochs')
        dim_hidden_size = skopt.space.Integer(low=512, high=2048, name='hidden_size')
        dim_learning_rate = skopt.space.Real(low=1e-6, high=1e-2, prior='log-uniform',
                                             name='learning_rate')
        dim_learning_rate_fine_tuning = skopt.space.Real(low=1e-6, high=1e-2, prior='log-uniform',
                                                         name='learning_rate_fine_tuning')
        dim_dropout = skopt.space.Real(low=0, high=0.9, name='dropout')
        dim_l2_lambda = skopt.space.Real(low=1e-6, high=1e-2, prior='log-uniform',
                                         name='l2_lambda')
        dim_fine_tuning = skopt.space.Categorical(categories=[True, False],
                                                  name='fine_tuning')

        dimensions = [dim_epochs,
                      dim_hidden_size,
                      dim_learning_rate,
                      dim_learning_rate_fine_tuning,
                      dim_dropout,
                      dim_l2_lambda,
                      dim_fine_tuning]

        # read default parameters from last optimization
        try:
            res = skopt.load('checkpoint.pkl')
            x0 = res.x_iters
            y0 = res.func_vals
            start_from_checkpoint = True
            print('Parameters of previous optimization loaded!')
        except:
            # fall back default values
            default_parameters = [2, 1024, 5e-4, 6e-4, 0.9, 1e-3, True]
            start_from_checkpoint = False


        checkpoint_saver = skopt.callbacks.CheckpointSaver('checkpoint.pkl', store_objective=False)
        checkpoint_downloader = training_utils.CheckpointDownloader('checkpoint.pkl')
        verbose = skopt.callbacks.VerboseCallback(n_total=num_iterations)

        @skopt.utils.use_named_args(dimensions=dimensions)
        def fitness(epochs, hidden_size, learning_rate, learning_rate_fine_tuning, dropout, l2_lambda, fine_tuning):
            """
            Function to be minimized by the optimization. Trains a model using the fit method and the given
            hyperparameters to return the final (negative) value of the validation accuracy.
            """
            # print the hyper-parameters
            logging.info('Fitnessing hyper-parameters')
            logging.info(f'epochs:{epochs}')
            logging.info(f'hidden_size:{hidden_size}')
            logging.info(f'learning rate:{learning_rate}')
            logging.info(f'learning rate fine tuning:{learning_rate_fine_tuning}')
            logging.info(f'dropout:{dropout}')
            logging.info(f'l2_lambda:{l2_lambda}')
            logging.info(f'fine_tuning:{fine_tuning}')

            # fit the model
            self.fit(epochs=epochs, hidden_size=hidden_size, learning_rate=learning_rate,
                     learning_rate_fine_tuning=learning_rate_fine_tuning,
                     dropout=dropout, l2_lambda=l2_lambda, fine_tuning=fine_tuning,
                     min_accuracy=min_accuracy)

            # extract fitness
            fitness = self.fitness

            logging.info(f'CALCULATED FITNESS OF:{fitness}')

            del self.model
            tf.keras.backend.clear_session()
            return -fitness

        # optimization
        if start_from_checkpoint:
            print('Continuous fitness')
            search_result = skopt.forest_minimize(func=fitness,
                                                  dimensions=dimensions,
                                                  x0=x0,  # already examined values for x
                                                  y0=y0,  # observed values for x0
                                                  # Expected Improvement.
                                                  acq_func='EI',
                                                  n_calls=num_iterations,
                                                  n_random_starts=n_random_starts,
                                                  callback=[checkpoint_saver, checkpoint_downloader, verbose])
        else:
            print('New fitness')
            search_result = skopt.forest_minimize(func=fitness,
                                                  dimensions=dimensions,
                                                  # Expected Improvement.
                                                  acq_func='EI',
                                                  n_calls=num_iterations,
                                                  n_random_starts=n_random_starts,
                                                  x0=default_parameters,
                                                  callback=[checkpoint_saver, checkpoint_downloader, verbose])

        if save_results:
            with open('hyperparameters_dimensions.pickle', 'wb') as f:
                dill.dump(dimensions, f)
            with open('hyperparameters_search.pickle', 'wb') as f:
                dill.dump(search_result.x, f)
            print("Hyperparameter search saved!")

        # build results dictionary
        results_dict = {dimensions[i].name: search_result.x[i]
                        for i in range(len(dimensions))}
        logging.info('Optimal fitness value of:{}'.format(-float(search_result.fun)))
        logging.info('Optimal hyperparameters:{}'.format(results_dict))


if __name__ == '__main__':
    classifier = ImageClassifier()
