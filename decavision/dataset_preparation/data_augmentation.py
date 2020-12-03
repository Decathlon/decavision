import os
import math
import shutil

import Augmentor


class DataAugmentor:
    """
    Class to generate augmented images for classification purposes. Images must be located in a folder with
    subfolders for each category.

    Arguments:
        path (str): location of the folders containing images for each category
        option (bool): use or not option during the augmentation, option can be anything between distortion,
            flip_horizontal, flip_vertical, random_crop, random_erasing, rotate, resize, skew, shear, brightness,
            contrast and color (options described at https://github.com/mdbloice/Augmentor)
        target_width (int): new width of resized images (if resized is True)
        target_height (int): new height of resized images (if resized is True)
    """
    
    def __init__(self, path, distortion=False, flip_horizontal=False,
                 flip_vertical=False, random_crop=False, random_erasing=False,
                 rotate=False, resize=False, skew=False,
                 shear=False, brightness=False, contrast=False,
                 color=False, target_width=299, target_height=299):
        self.path = path
        self.categories = sorted(os.listdir(self.path))
        self.categories_folder = [os.path.abspath(os.path.join(self.path, i)) for i in self.categories]
        # options
        self.distortion = distortion
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.random_crop = random_crop
        self.random_erasing = random_erasing
        self.rotate = rotate
        self.resize = resize
        self.skew = skew
        self.shear = shear
        self.brightness = brightness
        self.contrast = contrast
        self.color = color
        self.target_width = target_width
        self.target_height = target_height
        # Create a pipeline for each class
        self.pipelines = {}
        for folder in self.categories_folder:
            print("Folder %s:" % folder)
            self.pipelines[os.path.split(folder)[1]] = (Augmentor.Pipeline(folder))
            print("\n----------------------------\n")
            
        for p in self.pipelines.values():
            print("Class %s has %s samples." % (p.augmentor_images[0].class_label, len(p.augmentor_images)))
    
    def _set_options(self, pipeline):
        """
        Set desired options to the augmentation pipeline.

        Arguments:
            pipeline (obj): augmentor object to do data augmentation
        """
        if self.distortion: 
            pipeline.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
        if self.flip_horizontal: 
            pipeline.flip_left_right(probability=0.5)
        if self.flip_vertical: 
            pipeline.flip_top_bottom(probability=0.5)
        if self.random_crop: 
            pipeline.crop_random(probability=0.5, percentage_area=0.75)
        if self.resize: 
            pipeline.resize(probability=1, width=self.target_width, height=self.target_height, resample_filter="BILINEAR")
        if self.random_erasing: 
            pipeline.random_erasing(probability=0.5, rectangle_area=0.25)
        if self.rotate: 
            pipeline.rotate(0.5, max_left_rotation=10, max_right_rotation=10)
        if self.skew: 
            pipeline.skew(probability=0.5)
        if self.shear: 
            pipeline.shear(probability=0.25, max_shear_left=10, max_shear_right=10)
        if self.brightness: 
            pipeline.random_brightness(probability=0.25, min_factor=0.7, max_factor=1.3)
        if self.contrast: 
            pipeline.random_contrast(probability=0.25, min_factor=0.7, max_factor=1.3)
        if self.color: 
            pipeline.random_color(probability=0.25, min_factor=0.7, max_factor=1.3)
        
    def generate_images_single_class(self, class_size_approximation, category):
        """
        Generate augmented images for a single category according to the specified options. Images are generated
        until the desired size is reached and they are saved in an outputs folder at the location of the original
        images.

        Arguments:
            class_size_approximation (int): approximate total number of images wanted
            category (str): folder name where images to augment are located in the main folder
        """
        pipeline = self.pipelines[category]
        self._set_options(pipeline)
        for _ in range(math.ceil(class_size_approximation/len(pipeline.augmentor_images)-1)):
            pipeline.process()
        self._move_outputs(pipeline)
        self._delete_outputs(pipeline)
    
    def generate_images(self, class_size_approximation):
        """
        Generate augmented images for all categories in the main folder according to the specified options. Images
        are generated until the desired size is reached and they are saved in the same subfolder as the original
        images.

        Arguments:
            class_size_approximation (int): approximate total number of images wanted for each category
        """
        for pipeline in self.pipelines.values():
            self._set_options(pipeline)
            for _ in range(math.ceil(class_size_approximation/len(pipeline.augmentor_images)-1)):
                pipeline.process()
            self._move_outputs(pipeline)
            self._delete_outputs(pipeline)

    def _move_outputs(self, pipeline):
        """
        Move output folder content to main folder.

        Arguments:
            pipeline (obj): augmentor object to do data augmentation
        """
        output_dir = pipeline.augmentor_images[0].output_directory
        class_label = pipeline.augmentor_images[0].class_label

        for image in os.listdir(pipeline.augmentor_images[0].output_directory):
            os.rename(os.path.join(output_dir, image), os.path.join(os.path.join(self.path, class_label), image))

    def _delete_outputs(self, pipeline):
        """
        Delete output folder if present.

        Arguments:
            pipeline (obj): augmentor object to do data augmentation
        """
        dirpath = pipeline.augmentor_images[0].output_directory
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
