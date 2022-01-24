from flask import Flask, render_template,request

from sklearn.naive_bayes import MultinomialNB
from pythainlp.tokenize import word_tokenize
from pythainlp.tag.named_entity import ThaiNameTagger
from pythainlp.util import normalize
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
import time
import os
os.environ['PATH']
ner = ThaiNameTagger()


#import file data
File_name = 'trainingset_8100.csv'
#File_test = 'fakenews_testset.csv'
#File_test = 'testset_900.csv'
df = pd.read_csv(File_name,encoding='utf-8')
#dt = pd.read_csv(File_test,encoding='utf-8')
#File_9000 = 'dataset9000.csv'
#f9000 = pd.read_csv(File_9000,encoding='utf-8')


#declear variable for operation
#X = f9000['news']
#y = f9000['label']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
text = df['news'] 
#test = dt['news']
#input = []
text_clean = []
test_clean = []
text_cut = []
#test_cut = []
text_sent_cut = []
test_sent_cut = []
uniquify_text = []
uniquify_test = []
taged_text = []
taged_test = []
#USING same Variable
#text = X_train
#test = X_test
#loop for regular expression and word tokenize data
for i in range(len(text)):
    text_clean.append(re.sub(r'^https?:\/\/.*[\r\n]*', '', text.iloc[i], flags=re.MULTILINE))
    taged_text.append(ner.get_ner(text_clean[i]))
    for l in (word_tokenize(normalize(text_clean[i]), keep_whitespace=False, engine="newmm")):
        text_cut.append(l)
        if l not in uniquify_text:
            uniquify_text.append(l)
    text_sent_cut.append(text_cut)
    text_cut=[]

def input_tokenize(input_text):
    input= []
    test_cut = []
    input.append(input_text)
    for i in range(len(input)):
        test_clean.append(re.sub(r'^https?:\/\/.*[\r\n]*', '', input[i], flags=re.MULTILINE))
        taged_test.append(ner.get_ner(test_clean[i]))
        for l in (word_tokenize(normalize(test_clean[i]), keep_whitespace=False, engine="newmm")):
            test_cut.append(l)
            if l not in uniquify_test:
                uniquify_test.append(l)
        test_sent_cut.append(test_cut)
        test_cut=[]


label = le.fit_transform(df['label'])
#test_label = le.fit_transform(dt['label'])
#label = le.fit_transform(y_train)
#test_label = le.fit_transform(y_test)

#sklearn
def identity_tokenizer(text):
    return text
vectorizer = CountVectorizer(tokenizer=identity_tokenizer
                             ,vocabulary=uniquify_text
                             ,lowercase=False)

X_train_counts = vectorizer.fit_transform(text_sent_cut)
   
clf = MultinomialNB().fit(X_train_counts,label)
    


def predicted():
    X_new_counts = vectorizer.transform(test_sent_cut)
    predicted = clf.predict(X_new_counts)
    predict = ''
    if predicted[0] ==0:
        predict='มีแนวโน้มว่าจะเป็นข่าวปลอม'
    else:
        predict='ไม่มีแนวโน้มว่าจะเป็นข่าวปลอม'
    predicted = []
    return predict



app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("main.html")


#A POST request can include a query string, however normally it doesn't 
#a standard HTML form with a POST action will not normally include a query string for example.
@app.route("/result",methods=['POST'])
def result():
    #text = request.form['text']
    ''' 
    print(text)
    input_tokenize(text)
    predict = predicted()
    '''
    text ="ศูนย์ป้องกันน้ำท่วม รายงาน 19.00 น. กทม. ไม่พบกลุ่มฝน อุณหภูมิที่สำนักการระบายน้ำ 29 องศาเซลเซียส ความชื้นสัมพัทธ์ 65%"
    predict = 'มีแนวโน้มว่าจะเป็นข่าวปลอม'
    return render_template("result.html",recent_text=text,predict = predict)
