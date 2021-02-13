import cv2
import numpy as np
import random
import pafy  # library to work with youtube videos
from recognition import process_number, recognize
from split_plate import split_plate, dumb_split_plate

# set url to grab video from youtube
url = 'https://www.youtube.com/watch?v=Li3FPoa3iYE'
url1 ='https://www.youtube.com/watch?v=DWo1ESaLg-Q'
num_set = set()  # to store read data
jobs = []
t_config = '--user-patterns pattern.txt --oem 3 --psm 13 -c tessedit_char_whitelist=-BDFGHJKLNPRSTUVXYZ0123456789  load_system_dawg=false load_freq_dawg=false'

# start parameters for masking
h_min = 8
h_max = 21
s_min = 112 #152
s_max = 205
v_min = 75 #195
v_max = 215 #244
confi = 0.8 # i need confidence no less than
pause = False
show_mask = False


# function runs as parameters changed


def set_mask(_):
    global lower, upper
    lower = np.array([cv2.getTrackbarPos("Hue_min", "params"), cv2.getTrackbarPos("Sat_min", "params"),
                      cv2.getTrackbarPos("Val_min", "params")])
    upper = np.array([cv2.getTrackbarPos("Hue_max", "params"), cv2.getTrackbarPos("Sat_max", "params"),
                      cv2.getTrackbarPos("Val_max", "params")])
    return ()


# define possible area size of a license plate
lp_max = 1000
lp_min = 400

counter = 0
tilt = 0  # if license plates are tilted, we need to fix it, set angle here (positive if the road turns right)

video = pafy.new(url)
best = video.getbest(preftype="mp4")

capture = cv2.VideoCapture()
capture.open(best.url)
s, img = capture.read()

# let's select roi
roi = cv2.selectROI('select', img)
cv2.destroyWindow('select')  # closing roi select window manually

# creating window to adjust parameters
cv2.namedWindow("params")
cv2.resizeWindow("params", 400, 200)
cv2.createTrackbar("Hue_min", "params", h_min, 178, set_mask)
cv2.createTrackbar("Hue_max", "params", h_max, 178, set_mask)
cv2.createTrackbar("Sat_min", "params", s_min, 255, set_mask)
cv2.createTrackbar("Sat_max", "params", s_max, 255, set_mask)
cv2.createTrackbar("Val_min", "params", v_min, 255, set_mask)
cv2.createTrackbar("Val_max", "params", v_max, 255, set_mask)


# building mask
lower = np.array([cv2.getTrackbarPos("Hue_min", "params"), cv2.getTrackbarPos("Sat_min", "params"),
                  cv2.getTrackbarPos("Val_min", "params")])
upper = np.array([cv2.getTrackbarPos("Hue_max", "params"), cv2.getTrackbarPos("Sat_max", "params"),
                  cv2.getTrackbarPos("Val_max", "params")])

# cropping image to roi
img = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
min_diff = roi[2] * roi[3] / 5  # we skip the frame if it is changed in less pixels than min_diff

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
old_img_gray = np.copy(img_gray)


def t_recognize(img,c=0.3):  # function to ocr images
    save_ll=False
    #print ('splitting...')
    chars = dumb_split_plate(img)

    if save_ll:  # save pic
        for char in chars:
            cv2.imwrite('./ll/'+str(random.randint(0,1000000)) + ".png", char)


    if not chars:
        #print ('[main]:no chars')
        return False,0,0
    #print('[main]: returned chars:', len(chars))
    if len(chars) > 1:
        #print('processing...')
        nums = process_number(chars)
    else:
        #print ('less than one char')
        return False,0,0
    if len(nums) > 1:
        #print('recognizing...')
        try:
            t,conf = recognize(nums,c)

        except:
            #print('something went wrong')
            return False,0,0
        return t, conf, chars
    return False,0




# start main loop
while True:
    if not pause:  # if pause not pressed ('p' button)
        old_img_gray = np.copy(img_gray)
        s, img = capture.read()
        img = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k = np.sum(img_gray != old_img_gray)
        if k < min_diff:
            # skip if no significant changes
            continue

    if not s:  # no image captured - continue to next frame
        continue

    # processing image before searching for contours
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_blur = cv2.blur(img_hsv, (10, 10), 2)
    img_eroded = cv2.erode(img_blur, (30, 30), iterations=2)
    mask = cv2.inRange(img_eroded, lower, upper)

    if show_mask: cv2.imshow('mask', mask)

    # detecting contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = np.copy(img)  # we need a copy to draw on it and show
    for cont in contours:
        area = cv2.contourArea(cont)

        if lp_max > area > lp_min:  # checking if area of contour within defined range
            box = cv2.approxPolyDP(cont, 1, 1)
            x, y, w, h = cv2.boundingRect(box)
            if h * w < 100:  # if area is way too small
                continue
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 3)  # draw a box on copy

            number = img[y :y + h, x - 2:x + w]  # saving cropped license plate

            if 3 < (w / h) < 6:  # typical proportions of LP

                try:
                    cv2.imshow('params', number)
                except:
                    #print("error in contours detection, size is : ",number.shape)  # sometimes we get 0 in contour dimensions
                    continue
                if tilt != 0:  # fixing proportions if needed
                    pts1 = np.float32([[0, 0 + tilt], [number.shape[0], 0], [0, number.shape[1]],
                                       [number.shape[0], number.shape[1] - tilt]])
                    pts2 = np.float32(
                        [[0, 0], [number.shape[0], 0], [0, number.shape[1]], [number.shape[0], number.shape[1]]])

                    M = cv2.getPerspectiveTransform(pts1, pts2)
                    number = cv2.warpPerspective(number, M, (number.shape[1], number.shape[0]))
                text, conf, chars = t_recognize(number)  # run recognition

                if text and 3 < len(text) and conf > 0.6:
                    print('Detected license plate: ', text, "conf=",conf)
                    dummy=np.zeros((40,100,3))
                    if conf > confi:
                        c=int(conf*100)
                        cv2.putText(dummy, text, (0, 30), cv2.FONT_HERSHEY_PLAIN,1.3,(0,255,255),1 )
                        cv2.putText(dummy, str(c)+'%', (15, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                        cv2.imshow('detected',dummy)


    cv2.imshow("Contours", img_copy)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # save pic
        cv2.imwrite("car" + str(counter) + ".png", img)
    elif key == ord('q'):  # stop
        break
    elif key == ord('z') and chars:  # save chars
        for char in chars:
            cv2.imwrite ('./ll/'+str(random.randint(0,1000000))+'.png',char)
        print ('[main]: Saved...')
    elif key == ord('m'):  # turn mask window on/off
        if show_mask:
            cv2.destroyWindow('mask')
        show_mask = not show_mask
    elif key == ord('p'):  # set on pause
        pause = not pause

capture.release()
cv2.destroyAllWindows()
