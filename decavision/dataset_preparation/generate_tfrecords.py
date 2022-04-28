import csv
import math
import os

import tensorflow as tf

from decavision.utils import utils


class TfrecordsGenerator:
    """
    Class to transform images into tfrecords format to train neural networks. Resulting files can be
    saved to google storage or locally. Can't be used with a TPU because local files need to be read.
    Strongly inspired by: https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36
    """

    def __init__(self):
        pass

    def _to_tfrecord(self, img_bytes, label):
        """ Convert image and label to tfrecord example. """
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label))
        }))
        return example

    def convert_image_folder(self, img_folder='data/image_dataset/train',
                             output_folder='data/tfrecords_dataset/train',
                             multilabel=False,
                             img_folder_new=None, 
                             target_size=None,
                             shards=16):
        """
        Convert all images in a folder (like train or val) to tfrecords. Folder must contain subfolders for each category.
        Possibility to combine data from two folders to perform progressive learning. Tfrecords can be saved
        locally or on google storage. A csv file containing the names of the classes is also saved.

        For multilabel all the images must be in a single folder and their name should be of the form class1__class2__...__imagename.jpg
        so the classes will be extracted from the file names. To make sure no class is missed, all images in train/val/test are checked.

        Arguments:
            img_folder (str): location of the images
            output_folder (str): folder to save the results, content of folder is deleted to save new data
            img_folder_new (str): if specified, images from this folder are included in the tfrecords as
                new categories for the purpose of progressive learning
            multilabel (bool): True if it is a multilabel problem
            shards (int): number of files to create
            target_size (tuple(int,int)): size to reshape the images if desired
        """
        # Create output directory if it does not exists
        if not os.path.exists(output_folder):
            if not utils.is_gcs(output_folder):
                os.makedirs(output_folder)
                print('Created directory {}'.format(output_folder))
        else:
            utils.empty_folder(output_folder)

        # Get all file names of images present in folder
        if multilabel:
            classes = []
            for folder_name in ['train', 'val', 'test']:
                folder = os.path.join(os.path.dirname(img_folder), folder_name)
                if os.path.isdir(folder):
                    cls = [i.split('.')[0].split('__')[:-1] for i in os.listdir(folder)]
                    cls = [item for sublist in cls for item in sublist]
                    classes = classes + cls
            classes = sorted(list(set(classes)))
            img_pattern = os.path.join(img_folder, '*')
            nb_images = len(tf.io.gfile.glob(img_pattern))
            shard_size = math.ceil(1.0 * nb_images / shards)
        else:
            classes = sorted(os.listdir(img_folder))
            if img_folder_new:
                new_classes = sorted(os.listdir(img_folder_new))
                classes = classes + new_classes
            print(classes)
            img_pattern = os.path.join(img_folder, '*/*')
            if img_folder_new:
                img_pattern_new = os.path.join(img_folder_new, '*/*')
                img_pattern = [img_pattern, img_pattern_new]
            nb_images = len(tf.io.gfile.glob(img_pattern))
            shard_size = math.ceil(1.0 * nb_images / shards)
        print("Pattern matches {} images which will be rewritten as {} .tfrec files containing {} images each.".format(nb_images, shards, shard_size))

        def decode_jpeg_and_label(filename):
            bits = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(bits, channels=3)
            label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep=utils.check_sep())
            if multilabel:
                label = label.values[-1]
            else:
                label = label.values[-2]
            return image, label

        def resize_image(image, label):
            image = tf.image.resize(image, size=[*target_size])
            image = tf.reshape(image, [*target_size, 3])
            return image, label

        def recompress_image(image, label):
            image = tf.cast(image, tf.uint8)
            image = tf.image.encode_jpeg(image, quality=100, format='rgb',
                                         optimize_size=True, chroma_downsampling=False)
            return image, label

        AUTO = tf.data.experimental.AUTOTUNE

        filenames = tf.data.Dataset.list_files(img_pattern)  # This also shuffles the images
        dataset = filenames.map(decode_jpeg_and_label, num_parallel_calls=AUTO)
        if target_size:
            dataset = dataset.map(resize_image, num_parallel_calls=AUTO)
        dataset = dataset.map(recompress_image, num_parallel_calls=AUTO)
        dataset = dataset.batch(shard_size)  # sharding: there will be one "batch" of images per file

        print("Writing TFRecords")
        for shard_num, shard in enumerate(dataset):
            images, labels = shard
            images = images.numpy()
            labels = labels.numpy()
            # batch size used as shard size here
            shard_size = images.shape[0]
            # good practice to have the number of records in the filename
            filename = os.path.join(output_folder, "{:02d}-{}.tfrec".format(shard_num, shard_size))

            with tf.io.TFRecordWriter(filename) as out_file:
                for i in range(shard_size):
                    if multilabel:
                        label_list = [classes.index(x) for x in labels[i].decode('utf8').split('.')[0].split('__')[:-1]]
                    else:
                        label_list = [classes.index(labels[i].decode('utf8'))]
                    example = self._to_tfrecord(images[i],  # re-compressed image: already a byte string
                                                label_list)
                    out_file.write(example.SerializeToString())
                print("Wrote file {} containing {} records".format(filename, shard_size))

        # save classes locally or on gcs depending on the output folder
        if utils.is_gcs(output_folder):
            with open("classes.csv", "w", newline="") as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(classes)
            utils.upload_file_gcs(os.path.dirname(output_folder), "classes.csv")
            print("Saved classes to google storage")
        else:
            with open(os.path.join(os.path.dirname(output_folder), "classes.csv"), "w", newline="") as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(classes)
            print("Saved classes locally")


if __name__ == '__main__':
    transformer = TfrecordsGenerator()
