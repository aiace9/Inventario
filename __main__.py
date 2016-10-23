import numpy as np
import cv2


def soglia1(img):
    img_tmp = cv2.medianBlur(img, 5)
    # Countors recognition, 11 and 7 are experimental values, avrege
    return cv2.adaptiveThreshold(
        img_tmp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)


def soglia2(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


def structureSolver(base, hierarchy):
    base_loc = base
    parent = hierarchy[0][base_loc[len(base_loc) - 1]][3]
    if parent == -1:
        return base_loc
    else:
        base_loc.append(parent)
        structureSolver(base_loc, hierarchy)


cap = cv2.VideoCapture(0)
count = 0
while(True):
    count += 1
    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame = cv2.imread('qrcode.jpg')

    # to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = soglia1(gray)
    im2, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Serach one couour
    enc = []
    for i in range(len(hierarchy[0])):
        if hierarchy[0][i][2] == -1:
            tmp = []
            tmp.append(i)
            enc.append(tmp)
    # be carefull, the structureSolver change the array!!
    for i in enc:
        structureSolver(i, hierarchy)
    
    for i in enc:
        if len(i) > 4:
            cv2.drawContours(frame, contours[i[0]], -1, (255, 0, 0), 3)

    # Display the resulting frame
    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    cv2.imshow('frame', frame)
    cv2.imshow('frame2', im2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
