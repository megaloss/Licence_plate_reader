import cv2
import numpy as np




def split_plate(img):
    img = cv2.resize(img,(55,14))
    #img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #img_blur = cv2.blur(img_hsv, (10, 10), 2)
    #img_eroded = cv2.erode(img_blur, (30, 30), iterations=2)
    '''
    h_min = 8
    h_max = 21
    s_min = 112
    s_max = 205
    v_min = 75
    v_max = 215
    '''
    qr3 = np.quantile(img[:, :, 0], 0.8)
    #85 and 70 - best
    mask = cv2.inRange(img, (max(0,qr3-80),0,0), (255,255,225))
    mask=cv2.bitwise_not(mask)
    #img[:,:,0]=0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #threshold = 120
    #sh = np.min(img)
    #sh=len(np.unique(gray))
    #ch = int(np.mean(img))
    #blurred = cv2.blur(gray, (3, 3))
    #canny_output = cv2.Canny(blurred, threshold, threshold * 2)

    #binarized = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 0)
    cv2.imshow('binary', mask)

    contours, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if len (contours) < 5:
        return False
    #print ('found contours:', len(contours))
    numbers = []
    order = []

    for i, cont in enumerate(contours):

        box = cv2.approxPolyDP(cont, 1, 1)
        x, y, w, h = cv2.boundingRect(box)
        area = w * h
        if area > 170 and area < 20:
            continue
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        #if w*1.2 > h:
        #    continue
        if w < 2 or h < 5 or w > 9:
            continue
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        #if w <= 3:
        #
        x = x - 1
        w = w + 2
        #print('x=', x)
        x = max(0, x)
        h = min(img.shape[0],14)
        y=0
        #w=8
        #h = 10


        char = gray[y:y + h, x:x + w]

        numbers.append([char,x])
    numbers = sorted(numbers, key=lambda x: x[1])
    numbers = [number[0] for number in numbers]
    #cv2.putText(img, str(sh),(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255))
    cv2.imshow('sharpness', img)
    #for i in range(len(numbers)):
        #cv2.imshow (str(i),numbers[i])
    #print ('Returning from split ',len(numbers),'images')
    return numbers


def dumb_split_plate(img):
    img = cv2.resize(img, (65, 14))
    q=6
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    l_size = img.shape[1]//q
    h = min(img.shape[0],14)
    #print ('splitting image of size:', img.shape)
    contours=[]
    start=0
    arr=[]
    mean_green=np.mean(img[:,:,1])
    mean_red=np.mean(img[:,:,2])
    mean_gr=mean_green+mean_red
    for i in range (img.shape[1]):
        if np.mean(img[3:11,i,1])+np.mean(img[3:11,i,2])>mean_gr*1.14:
            img[:,i,0] = 255
            img[:, i, 1] = 0
            img[:, i, 2] = 0
            if i> start: arr.append([start, i-start])
            start=max(0,i-1)
    arr = sorted(arr, key=lambda x: x[1], reverse=True)[0:q]
    #print (arr[0:q])
    arr = sorted(arr, key=lambda x: x[0])
    for i,w in arr:
        w = min (w,9)
        if w<=5:
            i= max(0,i-1)
            w+=(8-w)
        else:
            w+=3
            i=max(0,i-1)
        contours.append(gray[0:h, i:i+w])
        #print ('i=',i,'w=',w)
        cv2.rectangle(img, (i, 0), (i+w,h), (255, 0, 0), 1)
        #print ((i, 0), (i+w,h))




    cv2.imshow('split', img)

    if len(contours) < 5: return False
    new_img = np.zeros((14,1))

    #for cont in contours:
        #new_img=np.hstack((new_img,cont))
    new_img = np.hstack(tuple(contours))
    cv2.imshow ('stack',new_img)
    #print ('Returning from split ',len(contours),'images each of size',contours[0].shape )
    return contours


