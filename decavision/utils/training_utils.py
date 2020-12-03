import os
import sys

import tensorflow.keras.backend as K
from tensorflow.keras.losses import categorical_crossentropy as logloss
from tensorflow.keras.losses import sparse_categorical_crossentropy as sparselogloss

try:
    from decavision.utils.colab_utils import authenticate_colab
except:
    pass


class CheckpointDownloader(object):
    """
    Class to download current state to Google Drive after each iteration of hyperparameter optimization.
    To be used as callback for scikit-optimize routine. Files are saved in a folder called Checkpoints.

    Arguments:
        checkpoint_path (str): location where the checkpoint files are;
    """

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

    def __call__(self, res):
        """
        If working on colab, upload checkpoint file to google drive.

        Arguments:
            res (scipy object): The optimization as a OptimizeResult object.
        """
        if 'google.colab' in sys.modules and os.path.exists(self.checkpoint_path):
            print('Uploading checkpoint ' + self.checkpoint_path + ' to Google Drive')
            drive = authenticate_colab()
            file_id = None
            # get the id of the Checkpoints folder and create it if it doesn't exist already
            file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
            for file in file_list:
                if file['title'] == "Checkpoints":
                    file_id = file['id']
            if not file_id:
                folder_metadata = {'title': 'Checkpoints', 'mimeType': 'application/vnd.google-apps.folder'}
                folder = drive.CreateFile(folder_metadata)
                folder.Upload()
                file_id = folder['id']

            file1 = drive.CreateFile({"title": "checkpoint.pkl",
                                      "parents": [{"kind": "drive#fileLink", "id": file_id}]})
            file1.SetContentFile(self.checkpoint_path)
            file1.Upload()


def custom_loss(old_logits, new_logits, old_classes, L=5, temp=5):
    """
    Distilling loss used when updating a model with new classes.
    It forces the model to remember what it learned about the old classes.
    Increasing the parameter L forces the model to remember more.
    High temperature puts more importance on the dominant classes and low temperature focuses on everything.

    Arguments:
        old_logits (keras tensor): classification layer of the old model, without the activation
        new_logits (keras tensor): classification layer of the new model, without the activation
        old_classes (int): number of classes in the old model
        L (float): parameter that controls how much information to remember
        temp (float): parameter that controls how many classes are important

    Returns:
        keras loss: loss function to be used during training
    """
    def loss(y_true, y_pred):
        y_soft = K.softmax(old_logits / temp)

        logits_pred = new_logits[:, :old_classes]
        y_pred_soft = K.softmax(logits_pred / temp)

        return sparselogloss(y_true, y_pred) + L * logloss(y_soft, y_pred_soft)

    return loss
