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
input_text = 'กรมอุตุนิยมวิทยา เผยพายุโซนร้อน โคนี อ่อนกำลังลงเป็นพายุดีเปรสชัน จะเคลื่อนขึ้นฝั่งบริเวณเมืองกวีนอน เวียดนาม ในเช้าวันนี้ (6 พ.ย.) ทำให้ภาคอีสานตอนล่างและภาคตะวันออก มีฝนปานกลาง-ลมแรง 6-7 พ.ย.นี้'
result = Function.preprocess_input_text(input_text)
predicted = new_model.predict(result)
predicted = preprocessing.binarize(predicted)

print(predicted)
