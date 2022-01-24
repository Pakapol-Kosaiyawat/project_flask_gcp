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
import pymongo

myclient = pymongo.MongoClient("mongodb+srv://warangkana_kh:Sadaharu123@cluster0.h4ueo.mongodb.net/fakenews_db?retryWrites=true&w=majority")
db = myclient['fakenews_db']
warning_news = db['warning_news']


warning_list = []
for i in warning_news.find():
        userdict = i
        warning_list.append(userdict)

import Function
new_model = tf.keras.models.load_model('my_model')

import twitter_scraping


def predicted(input_text):
    result = Function.preprocess_input_text(input_text)
    predicted = new_model.predict(result)
    predicted = preprocessing.binarize(predicted)
    result_binary = int(predicted[0][0])
   
    result_str = ''
    if result_binary ==0:
        result_str='มีแนวโน้มเป็นข่าวปลอม'
    else:
        result_str='ไม่มีแนวโน้มเป็นข่าวปลอม'
    
    return result_str

def search_related(input_text):
    search = twitter_scraping.detect_similarity(input_text)
    found = search[2]
    return found

app = Flask(__name__)

@app.route("/")
def hello_world():

    return render_template("main.html")

@app.route("/fastcheck")
def fast_check():
    return render_template("fastcheck.html")


#A POST request can include a query string, however normally it doesn't 
#a standard HTML form with a POST action will not normally include a query string for example.
@app.route("/result",methods=['POST'])
def result():
    text = request.form['text']
    predict = predicted(text)
    start_time = time.time()
    news_related = search_related(text)
    print("--- %s seconds ---" % (time.time() - start_time))
    return render_template("result.html",recent_text=text,predict = predict,related=news_related)

@app.route("/warning_news")
def warning_news(): 
    print(type(warning_list))
    return render_template("warning_news.html",warning_news = warning_list)

@app.route("/warning_news_detail/<id>")
def warning_news_detail(id):
    id = str(id)
    print(id)
    print(warning_list[0])
    for i in warning_list:
        if re.search(id,str(i)):
            print('news found')
            news_detail = i
    print(type(news_detail))
    news_detail = list(news_detail.items())
    print(type(news_detail))
    return render_template("warning_news_detail.html",all_detail = news_detail)

