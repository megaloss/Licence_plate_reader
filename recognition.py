import numpy as np
import cv2
from tensorflow import keras
#from sklearn.preprocessing import LabelEncoder
model = keras.models.load_model('model1.model')



def process_number(numbers):
    x_size = 14
    y_size = 14
    output = []
    #print ('len:', len (numbers))
    #for n in numbers:
    for n in numbers:

        data = np.array(n)/255
        # np.array(data.iloc[:, 0:-1]) / 255
        #print (data)

        if len(data) == 0:
            #print ('[recognition]: len(data)==0')
            continue
        #print ('[recognition]:processing shape:', data.shape)
        maxx = data.max()
        #print ('maxx=',maxx)
        #print ('size = ',data.shape)
        # padding array to the size 14*14 so all the images have the same size
        add_x = (x_size-n.shape[0])//2
        add_y = (y_size-n.shape[1])//2
        data = np.pad(data,((add_x,x_size-n.shape[0]-add_x),(add_y,y_size-n.shape[1]-add_y)),
                      mode='constant', constant_values=maxx)
        #data = data.flatten()
        data=data.reshape(1,196)
        output.append(data)
    #print ('returning', len (output), 'numbers from preprocessing, each of shape: ',output[0].shape)
    return output


#reading file with data (flattened array 14*14 + label at the end)
#data = pd.read_csv('chars1.csv')
#data = data.sample(frac=1)

#y = np.array (data.iloc[:,-1]).astype(str)
#encoder = LabelEncoder()
#y = encoder.fit_transform(y)

def recognize(inp,c=0.3):
    chars = '0123456789BDFGHJKLNPRSTVXZ'
    #print('recognition got input of type', type(inp),' of len:',len(inp), 'each has size', inp[0].shape, 'of type', type(inp[0]))
    #inp = inp.reshape(1,196)
    #cv2.imshow('letter', inp[0].reshape(14,14))
    #if len (inp)>2:
    #    cv2.imshow('letter1', inp[1].reshape(14, 14))
    #inp = np.zeros((1,196))
    #print(inp)
    result = ''
    #print('ready to predict')
    #print (len(inp))
    conf = 0
    min_c=1
    for i in inp:
        r = chars[np.argmax(model.predict(i))]
        result += r
        #cv2.imshow(r, i.reshape(14, 14))
        t=model.predict(i).max()
        conf+=t
        min_c = min(t,min_c)


    #print('predictions:', predictions)
    #for prediction in predictions:
    #    arg = np.argmax(prediction)
    #    result += chars[arg]
    #print(result)
    print('[recognition]: min_c=', min_c)
    #cv2.waitKey(0)
    if min_c > c:
        return result, conf/len(inp)
    else:
        return False, False


