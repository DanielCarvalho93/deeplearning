import pickle
import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img 
from keras.models import Sequential 
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense 
from keras import applications 
from keras.utils.np_utils import to_categorical 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math 
import datetime
import time
from tensorflow.keras.models import load_model

vgg16 = applications.VGG16(include_top=False, weights='imagenet') 
datagen = ImageDataGenerator(rescale=1. / 255)
model = load_model('model.h5')

final_table = pd.read_csv('finaldata.csv')
userdata = pd.read_csv('userdata.csv')


def read_image(path):
    print("[INFO] loading and preprocessing image...")  
    image = load_img(path, target_size=(224, 224))  
    image = img_to_array(image)  
    image = np.expand_dims(image, axis=0)
    image /= 255.  
    return image


def test_single_image(path):
    classes = ['MAGIC', 'MONSTERS', 'TRAP']
    images = read_image(path)
    time.sleep(.5)
    bt_prediction = vgg16.predict(images)  
    preds = model.predict(bt_prediction)
    label = []
    for idx, type, x in zip(range(0,6), classes , preds[0]):
        label.append(("{} {}%".format(type, round(x*100,2) )))
    return label


def ocr_reading(path):
    class_1 = test_single_image(path)
    reader = easyocr.Reader(['en'],gpu=False)
    result = reader.readtext(path)
    top_left = tuple(result[0][0][0])
    bottom_right = tuple(result[0][0][2])
    text = result[0][1]
    text = text.lower()
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(path)
    img = cv2.rectangle(img,top_left,bottom_right,[0,255,0],5)
    img = cv2.putText(img,text,top_left,font,1, (255,255,255),2,cv2.LINE_AA)
    cv2.imwrite('static/images/result.jpg', img)
    #name = text
    value = final_table[final_table['name'] == text]
    join_new = pd.concat([value,userdata], axis=0)
    join_new.to_csv('userdata.csv',index=False)
    return class_1, value