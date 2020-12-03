# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:36:37 2018

Main function to train an algorithm and-or classify images from a trained model

@author: AI team
"""
import argparse
import dill
import os

from decathlonian.dataset_preparation.extract_images import ExtractImages
from decathlonian.dataset_preparation.generate_tfrecords import TfrecordsGenerator
from decathlonian.model_testing.testing import ModelTester
from decathlonian.model_training.tfrecords_image_classifier import ImageClassifier
from decathlonian.utils.data_utils import split_train


# extract the arguments
parser = argparse.ArgumentParser(description='Extarct images, run hyperperameter search, fit the classifier, evalute the accuracy or predict the class')
parser.add_argument('--task', type=str, default='pass',
                    help="""
                    task to perform: 
                    extract_images-->build train, val and test sets from a Google Images search
                    split_training-->split a set of images into a training and a validation set
                    generate_tfrecords-->create tfrecords from image data
                    hyperparameters -->optimize classifier hyperparameters; 
                    fit-->fit the classifier (and optionaly save) the classifier; 
                    evaluate-->calculate the accuracy on a given set of images
                    classify-->predict the probability that the image is from the possible classes
                    """)
parser.add_argument('--save_model', type=int, default=0,
                    help="""
                    If we want (1) or not (0) to save the model we are fitting
                    """)
parser.add_argument('--extract_SavedModel', type=int, default=0,
                    help="""
                    If we want (1) or not (0) to save the model we are fitting in a production format
                    """)
parser.add_argument('--number_iterations', type=int, default=20,
                    help="""
                    Number of iterations to perform when doing a hyperparameter optimization. Has to be greater than parameter n_random_starts
                    """)
parser.add_argument('--n_random_starts', type=int, default=10,
                    help="""
                    Number of random combinations of hyperparameters to try first in the hyperparameters optimization
                    """)
parser.add_argument('--img', type=str, default=None,
                    help="""
                    Path of the images when we want to predict their classes
                    """)
parser.add_argument('--evaluate_directory', type=str, default='val',
                    help="""
                    If we want to evaluate accuracy based on images in the "train", "val" or "test" directory
                    """)
parser.add_argument('--batch_size', type=int, default=20,
                    help="""
                    Batch size of the classifier
                    """)
parser.add_argument('--transfer_model', type=str, default='Inception',
                    help="""
                    Base model used for classification - EfficientNet (B0 and B3), Inception (V3), Xception, Inception_Resnet (V2) and Resnet (50) currently supported
                    """)
parser.add_argument('--val_fraction', type=float, default=0.1,
                    help="""
                    Fraction of training images to move to the validation set
                    """)
args = parser.parse_args()

# ----------------------------------------------------------------------------------------------------------------------
# verify the format of the arguments

if args.task not in ['extract_images', 'hyperparameters', 'fit', 'evaluate', 'classify',
                     'split_training', 'generate_tfrecords', None]:
    print('Task not supported')
    args.task = 'pass'

if args.task == 'evaluate_directory':
    if args.evaluate_directory not in ['train', 'val', 'test']:
        print('evaluate_directory has to be train, val or test')
        args.task = 'pass'

if args.task == 'fit':
    if args.save_model == 1:
        save_model = 'model'
    elif args.save_model == 0:
        save_model = None
    else:
        print('save_model argument is not 0 or 1')
        args.task = 'pass'
    
    if args.extract_SavedModel == 1:
        extract_SavedModel = 'model'
    elif args.extract_SavedModel == 0:
        extract_SavedModel = None
    else:
        print('extract_SavedModel argument is not 0 or 1')
        args.task = 'pass'
    
if not (args.number_iterations > args.n_random_starts and isinstance(args.number_iterations, int)):
    print('number_iterations has to be an integer greater than 10')
    args.task = 'pass'
    
if not (args.batch_size > 0 and isinstance(args.batch_size, int)):
    print('batch_size has to be a positive integer')
    args.task = 'pass'

if args.task == 'classify':    
    if os.path.exists(args.img):
        img_path = args.img
    else:
        print('Unknown path')
        args.task = 'pass'
        
if args.transfer_model not in ['Inception', 'Xception', 'Resnet', 'Inception_Resnet', 'B0', 'B3', 'B5']:
    print(args.transfer_model + ' not supported. transfer_model supported: Inception, EfficientNet (B0, B3 and B5), Xception, Inception_Resnet and Resnet')
    args.task = 'pass'
else:
    if args.transfer_model in ['Inception', 'Xception', 'Inception_Resnet', 'B5', 'B3', 'B0']:
        target_size = (299, 299)
    else:
        target_size = (224, 224)

if args.task == 'split_training':        
    if not (args.val_fraction > 0 and isinstance(args.val_fraction, float) and args.val_fraction < 1):
        print('val_fraction has to be a float number between 0 and 1')
        args.task = 'pass'

# ----------------------------------------------------------------------------------------------------------------------
# functions to perform required tasks

def classify():
    categories = os.listdir('data/image_dataset/train')
    tester = ModelTester('model.h5')
    tester.classify_images(img_path, categories, False)

def extract_images():
    extracter = ExtractImages(output_dir='data', verbose=False)
    extracter.run()

def hyperparameters():
    classifier = ImageClassifier(tfrecords_folder='data/tfrecords',
                                 batch_size=args.batch_size,
                                 transfer_model=args.transfer_model)
    classifier.hyperparameter_optimization(num_iterations=args.number_iterations,
                                           n_random_starts=args.n_random_starts,
                                           save_results=True)

def split_training():
    split_train(split=args.val_fraction)

def generate_tfrecords():
    for folder in ['train', 'val']:
        shards = 1 + len(os.listdir('data/image_dataset/' + folder)) // 1000
        generator = TfrecordsGenerator()
        generator.convert_image_folder(img_folder='data/image_dataset/' + folder,
                                       output_folder='data/tfrecords/' + folder,
                                       target_size=target_size,
                                       shards=shards)

def fit():
    try:
        #read the optimized hyperparameters
        with open('hyperparameters_dimensions.pickle', 'rb') as f:
                dimensions = dill.load(f)
        with open('hyperparameters_search.pickle', 'rb') as f:
                sr = dill.load(f)
        opt_params = {dimensions[i].name:sr[i] for i in range(len(dimensions))}
        
    except:
        print('Could not find optimal hyperparameters. Selecting default values')
        opt_params = {}

    classifier = ImageClassifier(tfrecords_folder='data/tfrecords',
                                 batch_size=args.batch_size,
                                 transfer_model=args.transfer_model)
    classifier.fit(save_model=save_model,
                   export_model=extract_SavedModel,
                   **opt_params)

def evaluate():
    tester = ModelTester('model.h5')
    tester.evaluate('data/image_dataset/' + args.evaluate_directory)

# ----------------------------------------------------------------------------------------------------------------------
# run the proper function given the --task argument passed to the function

if args.task == 'hyperparameters':
    hyperparameters()

if args.task == 'generate_tfrecords':
    generate_tfrecords()
    
elif args.task == 'fit':
    fit()
    
elif args.task == 'extract_images':
    extract_images()
    
elif args.task == 'classify':
    classify()
    
elif args.task == 'evaluate':
    evaluate()
    
elif args.task == 'split_training':
    split_training()
