import cv2
import pandas as pd
import glob
import random
import numpy as np
import pickle
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
random.seed(10)

chars = '0123456789BDFGHJKLNPRSTVXZ'
x_size = 14
y_size = 14
z_size = 1
test_rate=0.02

def processFiles():

    output = pd.DataFrame()

    for char in chars:


        print (char)
        # image files are located in folders named as a character
        files = glob.glob('char/'+char+'/*.png')

        for file in files:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE ) # if an image is read not as grayscale we get a 3-D array !

            data = np.array(img)
            max=data.max()
            # padding array to the size 14*14 so all the images have the same size
            add_x = (x_size-img.shape[0])//2
            add_y = (y_size-img.shape[1])//2
            data = np.pad(data,((add_x,x_size-img.shape[0]-add_x),(add_y,y_size-img.shape[1]-add_y)) ,mode='constant',constant_values=max)
            data = data.flatten()


            # add a label to the end of the line
            data = np.append(data, char)

            # add a string to pandas dataframe
            output = output.append(pd.Series(data), ignore_index=True)

    #save dataframe to csv
    output.to_csv('chars1.csv',index=False)

# uncomment to re-read images into csv file
#processFiles()


#reading file with data (flattened array 14*14 + label at the end)
data = pd.read_csv('chars1.csv')
data = data.sample(frac=1)
# preprocessing data
X = np.array (data.iloc[:,0:-1])/255
y = np.array (data.iloc[:,-1]).astype(str)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

#split into testing and training data
split_at = int(len(X)*test_rate)
test_X = X[:split_at]
train_X = X[split_at:]
test_y = y[:split_at]
train_y = y[split_at:]

#building and compiling model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(x_size*y_size)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(len(chars), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#training the model
model.fit(train_X, train_y, epochs=60 , batch_size=20)
model.save('model1.model')
test_loss, test_acc = model.evaluate(test_X,  test_y, verbose=2)

print('Accuracy:', test_acc)
predictions = model.predict(test_X)


#visualisation of testing
for i in range(len(test_X)):
    img = np.uint8(test_X[i]*256)
    img = img.reshape((x_size, y_size))
    blank = np.ones((30,30,3))

    arg = np.argmax(predictions[i])
    #print (chars[arg])
    cv2.putText(blank,chars[arg],(7,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imshow('data', img)
    cv2.imshow('My guess', blank)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        print ('you pressed q')
        break
    


