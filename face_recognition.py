#!/usr/bin/env python
# coding: utf-8

# ## Real Time Face Recognition Project

# In[5]:


### Reading Video Using Webcam

## Prepared By : Mostafa Jubayer Khan
## Date : August 2, 2021

## Email: mostafajubayerkhan@gmail.com
## Linkedin : https://www.linkedin.com/in/mostafa-jubayer-khan/
## Github : https://github.com/mostafajubayerkhan

#################### Code Starts Here ###########################

import cv2
cap=cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    
    if ret == False:
        continue
    
    cv2.imshow("video frame",frame)
  
    key_pressed = cv2.waitKey(1) & 0xFF
    
    if key_pressed == ord('q'):
        break
# Close the Video Window (GUI)   
cap.release()
cv2.destroyAllWindows()


# In[7]:


## Recording & Saving Webcam Video Data for Recognizing the Person Later

## Prepared By : Mostafa Jubayer Khan
## Date : August 2, 2021

## Email: mostafajubayerkhan@gmail.com
## Linkedin : https://www.linkedin.com/in/mostafa-jubayer-khan/
## Github : https://github.com/mostafajubayerkhan

#################### Code Starts Here ###########################

import cv2
import numpy as np 

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = "./face_dataset/"

file_name = input("Enter the name of person : ")


while True:
	ret,frame = cap.read()

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret == False:
        continue

	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
	if len(faces) == 0:
        continue

	k = 1

	faces = sorted(faces, key = lambda x : x[2]*x[3] , reverse = True)

	skip += 1

	for face in faces[:1]:
		x,y,w,h = face

		offset = 5
		face_offset = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_selection = cv2.resize(face_offset,(100,100))

		if skip % 10 == 0:
			face_data.append(face_selection)
			print (len(face_data))


		cv2.imshow(str(k), face_selection)
		k += 1
		
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

	cv2.imshow("faces",frame)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print (face_data.shape)

# 
np.save(dataset_path + file_name, face_data)
print ("Dataset saved at : {}".format(dataset_path + file_name + '.npy'))

# Close the Video Window (GUI)
cap.release()
cv2.destroyAllWindows()


# In[8]:


## Final Part: Real Time Face Recognition using Webcam

## Prepared By : Mostafa Jubayer Khan
## Date : August 2, 2021

## Email: mostafajubayerkhan@gmail.com
## Linkedin : https://www.linkedin.com/in/mostafa-jubayer-khan/
## Github : https://github.com/mostafajubayerkhan

#################### Code Starts Here ###########################
import numpy as np
import cv2
import os

# K-Nearest Neighbor (KNN) Code in the following

def distance(v1, v2):
    # Eucledian 
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
    dist = []
    
    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]
    
    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]

## Capturing the Live Streaming Webcam Video

cap = cv2.VideoCapture(0)

# Haarcascade Classifier has been used 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_alt.xml")

dataset_path = "./face_dataset/"

face_data = []
labels = []
class_id = 0
names = {}


# Preparing the Dataset

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
print(face_labels.shape)
print(face_dataset.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

# Choosing the font for showing identified the Recognition Box 

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if ret == False:
        continue
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # To identify multi faces in an in image 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for face in faces:
        x, y, w, h = face

        # Reducing to the Region of Interest (ROI) of Image
        offset = 5
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())

        # Drawing a rectangle box surrounding the original image 
        cv2.putText(frame, names[int(out)],(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)

    cv2.imshow("Faces", frame)
    
    # Assigning 'Q' button to stop the GUI of WebCam Video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Close the Video Window (GUI)
cap.release()
cv2.destroyAllWindows()

