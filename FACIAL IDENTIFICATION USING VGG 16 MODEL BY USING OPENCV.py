##FACIAL IDENTIFICATION USING VGG 16 MODEL BY USING OPENCV
##TAKE THE IMAGES FROM THE FOLDERS AND STORE THE FACES IN SEPARATE FOLDER
import os,cv2
import numpy as np
from PIL import Image
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_classifier = cv2.CascadeClassifier('C:\\Users\\hp\\Downloads\\haarcascade_frontalface_default.xml')
img_paths=[]
count=0
main_dir="C:\\Users\\hp\\train"
for root,dirs,files in os.walk(main_dir):
    for direc in dirs:
        path=os.path.join(root,direc)
        img_paths.append(path)
o_p="C:\\Users\\hp\\face\\"
for i_p in img_paths:
    count=0
    for j in os.listdir(i_p):
        img_path=os.path.join(i_p,j)
        img_array=cv2.imread(img_path)
        gray = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
        faces=face_classifier.detectMultiScale(gray,1.3,5)
        count+=1
        for (x,y,w,h) in faces:
            cropped_face = img_array[y:y+h, x:x+w]
            direc=os.path.basename(i_p)
            face_dir=os.path.join(o_p,direc)
            if os.path.isdir(face_dir):
                cv2.imwrite(face_dir+"\\"+str(count)+".jpg",cropped_face)
                print(f"{direc} images are loading")
                print(count)
            else:
                os.mkdir(face_dir)
                cv2.imwrite(face_dir+"\\"+direc+"- "+ str(count) +".jpg",cropped_face)
                print(f"{direc} images are loading")
                print(count)
        continue



# VGG MODEL CREATION AND TRAINING
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE = [224, 224]

train_path = "C:\\Users\\hp\\train"

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

folders = glob("C:\\Users\\hp\\train\\*")

x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

r = model.fit_generator(
  training_set,
  epochs=5,
  steps_per_epoch=len(training_set)
)

plt.plot(r.history['loss'], label='train loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

plt.plot(r.history['accuracy'], label='train acc')
plt.legend()
plt.show()

from keras.models import load_model
os.mkdir("C:\\Users\\hp\\ann_mod")

model.save('C:\\Users\\hp\\ann_mod\\facefeatures_new_model.h5')



# RECOGNITION AND PUT A PREDICTED TEXT ON THE FACE
import cv2
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np

import os

nm_l=[]
main_dir="C:\\Users\\hp\\train"
for root,dirs,files in os.walk(main_dir):
        for i in dirs:
            nm_l.append(i)

model=load_model('C:\\Users\\hp\\ann_mod\\facefeatures_new_model.h5')

face_classifier = cv2.CascadeClassifier('C:\\Users\\hp\\Downloads\\haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]    
        roi_color = img[y:y+h, x:x+w]
        face = cv2.resize(roi_color, (224, 224))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im) 
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        ind = np.argmax(pred)
        label_name=nm_l[ind]
#         print(label_name)
        font=cv2.FONT_HERSHEY_PLAIN
        color=[255,0,255]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.putText(img,label_name,(x,y+50), font, 3, color, 3)
    cv2.imshow('img',img)
    cv2.imshow("gray",gray)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
