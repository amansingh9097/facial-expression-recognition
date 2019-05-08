# coding: utf-8

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


with open('dataset/fer2013.csv','r') as file:
    contents = file.readlines()
    
lines = np.array(contents)

no_of_instances = lines.size
print("no. of instances: ", no_of_instances)


print("instance length: ",len(lines[1].split(",")[1].split(" ")))


X_train, y_train, X_test, y_test = [], [], [], []

for i in range(1, no_of_instances): # starting from 2nd line bcz 1st line is header
    emotion, pixels, usage = lines[i].split(",")
    
    val = pixels.split(" ")
    img_px = np.array(val, 'float32')
    
    emotion = keras.utils.to_categorical(emotion, 7) 
    """
    this above step is crucial, even though the emotion column already had categorical values,
    a mismatch of dimension in output layer occurs if I don't explicitly convert them to categorical.
    Or may be earlier the values were as string. Need to find out!
    For now changing emotion to categorical values works.
    """
    
    if 'Training' in usage:
        X_train.append(img_px)
        y_train.append(emotion)
    elif 'PublicTest' in usage:
        X_test.append(img_px)
        y_test.append(emotion)


#data transformation for train and test sets
X_train = np.array(X_train, 'float32')
y_train = np.array(y_train, 'float32')
X_test = np.array(X_test, 'float32')
y_test = np.array(y_test, 'float32')

X_train /= 255 #normalize inputs between [0, 1]
X_test /= 255

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_train = X_train.astype('float32')
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
X_test = X_test.astype('float32')

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


#initializing the CNN
classifier = Sequential()

#1st convolution layer
classifier.add(Conv2D(64,(5,5), activation='relu', input_shape=(48,48,1)))
classifier.add(MaxPooling2D(pool_size=(5,5), strides=(2,2)))

#2nd convolution layer
classifier.add(Conv2D(64,(3,3), activation='relu'))
classifier.add(Conv2D(64,(3,3), activation='relu'))
classifier.add(AveragePooling2D(pool_size=(3,3), strides=(2,2)))

#3rd convolution layer
classifier.add(Conv2D(128,(3,3), activation='relu'))
classifier.add(Conv2D(128,(3,3), activation='relu'))
classifier.add(AveragePooling2D(pool_size=(3,3), strides=(2,2)))

#Flattening
classifier.add(Flatten())

#Full connection
classifier.add(Dense(units=1024, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1024, activation='relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units=7, activation='softmax'))


train_datagen = ImageDataGenerator()
training_set = train_datagen.flow(X_train, y_train, batch_size=256)


#Compile the CNN
classifier.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

classifier.fit_generator(
    training_set,
    steps_per_epoch=256,
    epochs=10
)

train_score = classifier.evaluate(X_train, y_train, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', 100*train_score[1])
 
test_score = classifier.evaluate(X_test, y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', 100*test_score[1])


#testing on custom images
img = image.load_img("D:\My_Work\PyCharm\Facial-Expression-Recognition\dataset\images\example4.png", grayscale=True, target_size=(48, 48))
 
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
 
x /= 255
 
custom = classifier.predict(x)

emotion_analysis(custom[0])
 
x = np.array(x, 'float32')
x = x.reshape([48, 48]);
 
plt.gray()
plt.imshow(x)
plt.show()


def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    plt.show()

