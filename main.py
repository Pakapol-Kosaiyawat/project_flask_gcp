from flask import Flask, render_template,request
import pandas as pd
import numpy as np
import time
from pythainlp.tokenize import word_tokenize
import re
import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from pythainlp.tag.named_entity import ThaiNameTagger
ner = ThaiNameTagger()
from sklearn import preprocessing



import Function

new_model = tf.keras.models.load_model('my_model')


def predicted(input_text):
    result = Function.preprocess_input_text(input_text)
    predicted = new_model.predict(result)
    predicted = preprocessing.binarize(predicted)
    result_binary = int(predicted[0][0])

    result_str = ''
    if result_binary ==0:
        result_str='มีแนวโน้มเป็นข่าวปลอม'
    else:
        result_str='มีแนวโน้มเป็นข่าวปลอม'
    
    return result_str



app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("main.html")


#A POST request can include a query string, however normally it doesn't 
#a standard HTML form with a POST action will not normally include a query string for example.
@app.route("/result",methods=['POST'])
def result():
    text = request.form['text']
    predict = predicted(text)
    

    return render_template("result.html",recent_text=text,predict = predict)
