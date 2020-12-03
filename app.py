# -*- coding: utf-8 -*-
"""
Router for the endpoint of the image classifier

For a detailed implementation example, see: https://blog.keras.io/index.html

@author: AI team
"""
import argparse
import io
import os

from flask import Flask, request, jsonify
from flask_cors import CORS

from decathlonian.utils.data_utils import prepare_image
from decathlonian.utils.utils import load_model

app = Flask(__name__)
CORS(app)

# extract the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='data/trained_models/trained_model.h5',
                    help='Location of the model to load, including its name')
parser.add_argument('--data_path', type=str, default='data/image_dataset/val',
                    help='Location of the data used for training. Necessary to get categories.')
args = parser.parse_args()

model_path = args.model_path
data_path = args.data_path

# function to load keras model
def prepare_model():
    global classes
    global target_size
    global model
    model = load_model(model_path)
    target_size = model.input_shape[1:3]
    classes = list(os.walk(data_path))[0][1]

@app.route('/classify', methods=['POST'])
def classify():
    # initialize the returned data dictionary
    data = {"success": False}
    
    if request.method == "POST":
        if request.files.get("img"):
            # read the image and preprocess it
            image = request.files["img"].read()
            image = prepare_image(io.BytesIO(image), target_size)

            # classify the image
            results = model.predict(image)[0]
            data["predictions"] = []

            # loop over the results and add them to returned dictionary
            for i in range(len(classes)):
                r = {"label": classes[i], "probability": float(results[i])}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data as a JSON
    return jsonify(data)
    
if __name__ == '__main__':
    print('Loading classification model')
    prepare_model()
    app.run()