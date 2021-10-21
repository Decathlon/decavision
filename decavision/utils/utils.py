import csv
import os
import sys

from google.cloud import storage
import tensorflow as tf
import tensorflow_hub as hub


def load_model_clear(path, include_top=True):
    """
    Clear tensorflow session and load keras .h5 model.

    Arguments:
        path (str): location of the model, including its name
        include_top (bool): whether or not to include the last layer of the model

    Returns:
        tf.keras model: loaded model
    """
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(path, compile=include_top, custom_objects={"KerasLayer": hub.KerasLayer})
    print('Model loaded !')
    if not include_top:
        model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    return model


def check_PU():
    """
    Check if machine is running on TPU, GPU or CPU.
    
    Returns:
        bool: whether or not the machine runs on a TPU
        bool: whether or not the machine runs on a GPU
    """
    use_tpu = False
    use_gpu = False
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
        use_tpu = True
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])  
    except ValueError:
        if tf.test.is_built_with_cuda():
            use_gpu = True
            print("Running on GPU", tf.test.gpu_device_name())
        else:
            print("Running on CPU")
    return use_tpu, use_gpu


def check_sep():
    """
    Check if the OS is windows or anything else to return the right separator.

    Returns:
         str: '\\' for windows and '/' for others.
    """
    if sys.platform == 'win32':
        return '\\'
    else:
        return '/'


def gcs_bucket(folder):
    """
    Create bucket object to be used to access files in google could storage bucket.

    Arguments:
        folder (str): name of the google storage folder, must be of the form gs://bucketname/prefix

    Returns:
        bucket object: google storage object
        string: prefix of the bucket, to be used to access files
    """
    path_split = folder.split("/")
    bucket_name = path_split[2]
    prefix = os.path.join(*path_split[3:])
    storage_client = storage.Client(project=None)
    bucket = storage_client.get_bucket(bucket_name)
    return bucket, prefix


def empty_folder(folder):
    """
    Delete all files in a given folder. First try to delete
    locally and if it fails try to delete in google cloud
    storage. If folder is in GCS, the link must include the
    gs:// part and the folder will be deleted as well.

    Arguments:
        folder (str): path of folder where files are located
    """
    if is_gcs(folder):
        bucket, prefix = gcs_bucket(folder)
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            blob.delete()
        print("Deleted files in gcs directory", folder)
    else:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            os.remove(file_path)
        print("Deleted files in local directory", folder)


def upload_file_gcs(gcp_path, file_path):
    """
    Copy local file to a folder in google cloud storage and call it classes.csv.

    Arguments:
        gcp_path (str): full path to gcp folder
        file_path (str): location of path to upload
    """
    bucket, prefix = gcs_bucket(gcp_path)
    blob = bucket.blob(prefix + '/classes.csv')
    blob.upload_from_filename(file_path)


def is_gcs(path):
    """
    Check if path is to a google cloud storage directory or a local one. Determined from the presence of 'gs'
    at the beginning of the path.

    Arguments:
        path (str): path to assess

    Returns:
         bool: True if path is on gcs and False if local
    """
    if path[:2] == "gs":
        return True
    else:
        return False


def load_classes(path, name='classes'):
    """
    Open csv and create list from its content. If csv is on google cloud storage,
    the file is downloaded into working directory and then opened.

    Arguments:
        path (str): location of the csv file
        name (str): name of csv file to open, without extension

    Returns:
        list: first line of the csv file"""
    if is_gcs(path):
        bucket, prefix = gcs_bucket(path)
        for blob in bucket.list_blobs(prefix=prefix + '/' + name + '.csv'):
            blob.download_to_filename(name + ".csv")
        with open('classes.csv', newline='') as f:
            reader = csv.reader(f)
            classes = next(reader)
    else:
        with open(os.path.join(path, name + '.csv'), newline='') as f:
            reader = csv.reader(f)
            classes = next(reader)
    return classes
