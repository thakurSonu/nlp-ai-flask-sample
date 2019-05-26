# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle

from utils import spacy_tokenizer
from utils import predictors

import pandas as pd


app = Flask(__name__)


# Load the model
modelFile = open('model.pkl','rb')
#modelFile.seek(0)
model = pickle.load(modelFile)


@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    print('data', data)

    testDataframe = pd.DataFrame([data['text']])


    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(testDataframe.iloc[0])

    # Take the first value of prediction
    output = prediction[0]

    response = { "prediction": str(output)}

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
