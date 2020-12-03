import os

try:
    from google.colab import auth
    from oauth2client.client import GoogleCredentials
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
except:
    pass
from pyunpack import Archive


def authenticate_colab():
    """
    Ask the user to connect to his google account to access google drive and google storage.

    Returns:
        google drive object
    """
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    return drive


def download_dataset(file_id, save_path, drive):
    """
    Download compressed dataset (zip or 7z format) from google drive and extract it.

    Arguments:
        file_id (str): id of the file to download
        save_path (str): location where data is extracted
        drive (google drive object): return of authenticate_colab function
    """
    file = drive.CreateFile({'id': file_id})
    filename = file['title']
    print('Downloading {}'.format(filename))
    file.GetContentFile(filename)
    print('File {} downloaded'.format(filename))
    print('Extracting images')
    try:
        Archive(filename).extractall(save_path)
        print('Images extracted')
        os.remove(filename)
    except Exception as e:
        print('Could not extract images because of {}'.format(e))
