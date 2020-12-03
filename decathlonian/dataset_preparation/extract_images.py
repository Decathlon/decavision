import argparse
import csv
import os
from os import path
import shutil
import urllib

import pandas as pd
from selenium import webdriver


class ExtractImages:
    """
    Class to search words on google images and download a specific number of images to
    make a dataset for image classification. A csv file with the following columns must be
    included in the output_dir: search_term, number_imgs, category. The file has to be named
    searchterms.csv.

    Arguments:
        output_dir (str): directory where images will be downloaded and where the csv file is located
        verbose (bool): print a line whenever an image is downloaded
    """

    def __init__(self, output_dir='google_data', verbose=True):
        self.output_dir = output_dir
        self.image_dir = os.path.join(self.output_dir, 'image_dataset', 'train')
        self.output_path_csv = os.path.join(os.getcwd(), self.output_dir, 'google_image_records.csv')
        self._append_new_line_to_csv(['search_term', 'image'])
        self.verbose = verbose

    def _append_new_line_to_csv(self, list_to_write):
        """
        Append a list as a new line to a csv located at output_path_csv

        Arguments:
            list_to_write (List[str]): list that is added to the csv
        """
        with open(self.output_path_csv, 'a') as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow(list_to_write)

    def _delete_images(self, folder):
        """
        Delete all the content of a folder.

        Arguments:
            folder (str): folder to delete the content of
        """
        if os.path.exists(folder):
            for the_file in os.listdir(folder):
                file_path = os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)

    def _get_images(self, search_term, nb_images, folder_name, browser):
        """
        Extract images on google for a given search term.

        Arguments:
            search_term (str): word to search on google
            nb_images (int): maximum number of images to download
            folder_name (str): folder where the images are saved
            browser (selenium browser object): browser to use to search on google"""
        print('Searching for {}'.format(search_term))
        url = 'https://www.google.co.in/search?q={}&source=lnms&tbm=isch'.format(search_term)
        browser.get(url)

        # we roll through the window, to be able to extract more than 100 images if desired
        for _ in range(10000):
            browser.execute_script("window.scrollBy(0,10000)")

        success = 0
        html = browser.page_source.split('["')
        for i in html:
            if i.startswith('http') and i.split('"')[0].split('.')[-1] in ['jpg', 'jpeg']:
                img_url = i.split('"')[0]
                img_name = os.path.basename(img_url)
                try:
                    urllib.request.urlretrieve(img_url, os.path.join(folder_name, img_name))
                    if self.verbose:
                        print('Saved image name {}'.format(img_name))
                    success = success + 1
                    self._append_new_line_to_csv([search_term, img_name])
                except:
                    pass

            if success >= nb_images:
                break

        print('{} pictures successfully downloaded for {}\n'.format(success, search_term))
        browser.close()

    def _organize_images(self):
        """
        Take all the images in the different folders for a given category and combine them
        into a single folder for each category.
        """
        for category in os.listdir(self.image_dir):
            category_path = os.path.join(self.image_dir, category)
            for search_word in os.listdir(category_path):
                search_word_path = os.path.join(category_path, search_word)
                for image in os.listdir(search_word_path):
                    image_path = os.path.join(search_word_path, image)
                    if image not in os.listdir(category_path):
                        shutil.move(image_path, category_path)
                shutil.rmtree(search_word_path)
            print("Images in {} moved".format(category))

    def run(self, delete_previous_images=False, path_to_driver=None):
        """
        Download images from google based on the search terms mentioned in the csv file.
        Each word is searched for the desired number of images and these are saved in a folder
        located at output_dir/image_dataset/train/category. A file called 'google_image_records.csv'
        is also created in the output directory with a list of all downloaded images.

        Arguments:
            delete_previous_images (bool): if we want to empty the directory before the search, to avoid duplicate images,
                the whole image_dataset folder will be emptied
            path_to_driver (str): path to the chrome driver; if None, we assume it is in the output_folder
        """
        # read the search terms csv
        search_terms_df = pd.read_csv(os.path.join(self.output_dir, 'searchterms.csv'),
                                      delimiter=',', encoding='latin1')
        search_terms_df['search_term'] = search_terms_df['search_term'].str.lower()
        search_terms = sorted(search_terms_df['search_term'].unique())

        # if we want to delete previously extracted images, to avoid duplicates
        if delete_previous_images:
            print('Deleting previous images')
            # training set
            self._delete_images(self.image_dir)

        # create the name of the folders
        for name in search_terms_df['category'].unique():
            sport_folder = path.join(self.image_dir, name)
            if not os.path.exists(sport_folder):
                os.makedirs(sport_folder)

        # extract the images
        for index, search_term in enumerate(search_terms):
            # we launch Chrome using selenium
            # if the path of the chrome driver is provided, we use it;
            # otherwise, we assume it is in the output directory
            if path_to_driver:
                browser = webdriver.Chrome(path_to_driver)
            else:
                browser = webdriver.Chrome(os.path.join(self.output_dir, 'chromedriver.exe'))

            header = {'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64)"
                                  "AppleWebKit/537.36 (KHTML, like Gecko)"
                                  "Chrome/43.0.2357.134 Safari/537.36"}

            nb_images = search_terms_df.loc[
                search_terms_df.search_term == search_term,
                'number_imgs'].values[0]
            sport_folder = search_terms_df.loc[
                search_terms_df.search_term == search_term,
                'category'].values[0]

            folder_name = path.join(self.image_dir, sport_folder, search_term)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            self._get_images(search_term, nb_images=nb_images,
                             folder_name=folder_name,
                             browser=browser)
            del browser
        self._organize_images()


def main():
    parser = argparse.ArgumentParser(
        description='Scrap image from google image search'
    )
    parser.add_argument('--output_dir', type=str,
                        default=path.join(path.dirname(__file__), '../google_data'),
                        help='Folder to save scrapped images')
    parser.add_argument('--verbose', '-v', dest='verbose',
                        default=True, type=bool,
                        help='Verbose')
    args = parser.parse_args()

    # extract search terms
    extracter = ExtractImages(args.output_dir, verbose=args.verbose)
    extracter.run()


if __name__ == '__main__':
    main()
