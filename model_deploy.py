# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:14:37 2021

@author: WiN 10
"""
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
import re
import time
import os
os.environ['PATH']
ner = ThaiNameTagger()

#import file data
File_name = 'trainingset_8100.csv'
#File_test = 'fakenews_testset.csv'
File_test = 'testset_900.csv'
df = pd.read_csv(File_name,encoding='utf-8')
dt = pd.read_csv(File_test,encoding='utf-8')
#File_9000 = 'dataset9000.csv'
#f9000 = pd.read_csv(File_9000,encoding='utf-8')


#declear variable for operation
#X = f9000['news']
#y = f9000['label']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
text = df['news'] 
test = dt['news']
#test = ['นพ.สิทธิพงศ์ ฬาพานิช น้องชายนอ.หญิง พร. พญ.สรัญญา ฬาพานิช แพทย์รพ.ภูมิพลที่เสียชีวิตจากโควิด โพสต์เฟซบุ๊กระบุ ครอบครัวสูญเสียจากโควิดแล้ว 3 ชีวิต พี่สาวที่ฉีดวัคซีนโควิดแล้วยังเสียชีวิต วอนเร่งนำวัคซีนคุณภาพเข้าประเทศ ']
text_clean = []
test_clean = []
text_cut = []
test_cut = []
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

for i in range(len(test)):
    test_clean.append(re.sub(r'^https?:\/\/.*[\r\n]*', '', test.iloc[i], flags=re.MULTILINE))
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
    
X_new_counts = vectorizer.transform(test_sent_cut)
predicted = clf.predict(X_new_counts)
print(predicted)


    
'''

print("MultinomialNB")
print("Number of mislabeled points out of a total %d points : %d"% (len(test_label), (predicted != test_label).sum()))
print('Accuracy : ',1 - (predicted != test_label).sum()/len(test_label))
tn, fp, fn, tp= confusion_matrix(test_label,predicted).ravel()
print('tn, fp, fn, tp = ',tn, fp, fn, tp)
print('confusion_matrix\n',confusion_matrix(test_label,predicted))
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2*precision*recall/(precision+recall)
print("F1 score : ",f1)
print('______________________________________________________')

def cross_valid():
    def identity_tokenizer(text):
        return text
    vectorizer = CountVectorizer(tokenizer=identity_tokenizer
                                 ,vocabulary=uniquify_text
                                 ,lowercase=False)
    
    X_train_counts = vectorizer.fit_transform(text_sent_cut)
    
    clf = MultinomialNB().fit(X_train_counts,label)
        
    print("Cross Validation MultinomialNB")
    #cross validation
    print('score:',cross_val_score(clf, X_train_counts,label,cv = 10))
    print('Sum Score: ',cross_val_score(clf, X_train_counts,label,cv = 10).sum()/len(cross_val_score(clf, X_train_counts,label,cv = 10)))
    
cross_valid()

start = time.time()
predicted = clf.predict(X_new_counts)
end = time.time()
print('Time to predict: ',end - start,'s')
'''