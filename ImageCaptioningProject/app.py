
import numpy as np
import os
import sys
import time
import glob
import re

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, send_file

app=Flask(__name__)
model_path='vgg19.h5'
model = load_model(model_path)
model.make_predict_function()

## Preprocessing function
def model_predict(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return predictions

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        img = request.files['image']
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(basepath,'uploads',img.filename)
        img.save(file_path)
        predictions = model_predict(file_path, model)
        pred_class=decode_predictions(predictions,top=1)
        result=str(pred_class[0][0][1])
        return result
    return None
    
if __name__ == '__main__':
    app.run(debug=True)
