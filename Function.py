# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:46:48 2021

@author: WiN 10
"""
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

File_9000 = 'dataset9000.csv'
f9000 = pd.read_csv(File_9000,encoding='utf-8')
x = f9000['news']
y = f9000['label']
test_file = 'test10new.csv'
test = pd.read_csv(test_file,encoding='utf-8')

def preprocess(text):#input parameter as a list of text
    text_clean = []
    text_cut = []
    text_sent_cut = []
    uniquify_text = []
    taged_text = []
    encoded_text = []
    max_word = 300

    for i in range(len(text)):
        text_clean.append(re.sub(r'^https?:\/\/.*[\r\n]*', '', text.iloc[i], flags=re.MULTILINE))
        taged_text.append(ner.get_ner(text_clean[i]))
        for l in (word_tokenize(text_clean[i], keep_whitespace=False, engine="newmm")):
            text_cut.append(l)
            if l not in uniquify_text:
                uniquify_text.append(l)
        text_sent_cut.append(text_cut)
        text_cut = []

    #pre-processing
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        output_mode='int',
        split=None,
        vocabulary=uniquify_text)    
    #encode input
    for i in range(len(text_sent_cut)):
        t1 = np.array(encoder(text_sent_cut[i]))
        t1 = np.reshape(t1,len(t1))
        t2 = np.zeros(max_word, dtype=int)
        t2[0:len(t1)] = t1
        encoded_text.append(t2)
        
    return tf.stack(encoded_text),encoder

#create endoced data and encoder
data,encoder = preprocess(x)

def preprocess_input_text(text,encoder=encoder):#input parameter as a list of text
    text_cut = []
    text_sent_cut = []
    max_word = 300

    for l in (word_tokenize(text, keep_whitespace=False, engine="newmm")):
        text_cut.append(l)
    text_sent_cut.append(text_cut)
    text_cut = []
 
    #encode input
    for i in range(len(text_sent_cut)):
        t1 = np.array(encoder(text_sent_cut[i]))
        t1 = np.reshape(t1,len(t1))
        encoded = np.zeros(max_word, dtype=int)
        encoded[0:len(t1)] = t1
        
    return tf.stack([encoded])
#crete encoded single input with trainset encoder
def preprocess_label(label):#input as binarize data
    LabelEncoded = []
    for i in label:
        LabelEncoded.append([int(i)])
    return tf.stack(LabelEncoded)
#create encoded label
#label = preprocess_label(y)

def create_model(encoder):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])
    model.summary()
    return model
#create model use to predict data
#model = create_model(encoder)
#model.fit(data,label,batch_size=64,epochs=100)

'''
input_text = 'ศูนย์ป้องกันน้ำท่วม รายงาน 19.00 น. กทม. ไม่พบกลุ่มฝน อุณหภูมิที่สำนักการระบายน้ำ 29 องศาเซลเซียส ความชื้นสัมพัทธ์ 65%'
result = preprocess_input_text(input_text,encoder)
model.predict(result)

print(result)
print(type(result))
'''

#to save and load model
#model.save('my_model')