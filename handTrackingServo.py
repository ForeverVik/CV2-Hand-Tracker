import cv2
import mediapipe as mp
import numpy as np
import time
import pyfirmata

#640x480
#1280x720
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

cap = cv2.VideoCapture(0)
cap.set(3,SCREEN_WIDTH)
cap.set(4,SCREEN_HEIGHT)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
print(cap.get(3))

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

lmX = []
lmY = []

leftHandCenterX = 0
leftHandCenterY = 0

rightHandCenterX = 0
rightHandCenterY = 0

board = pyfirmata.Arduino("/dev/cu.usbmodem14201")
servo = board.get_pin('d:10:s')

it = pyfirmata.util.Iterator(board)
it.start()

servo.write(0)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    bgImg = np.zeros((SCREEN_HEIGHT,SCREEN_WIDTH,3), np.uint8)

    shownImg = bgImg

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = shownImg.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)

                lmX.append(cx)
                lmY.append(cy)

                if id == 9:
                    leftHandCenterX = cx
                    leftHandCenterY = cy
                elif id == 29:
                    rightHandCenterX = cx
                    rightHandCenterY = cy

            mpDraw.draw_landmarks(shownImg, handLms, mpHands.HAND_CONNECTIONS)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #cv2.putText(shownImg, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image BlackBG", shownImg)
    cv2.imshow("Image", img)

    if len(lmX) > 0:
        if lmX[0] > 0:
            servo.write(lmX[0]/SCREEN_WIDTH*180)


    lmX = []
    lmY = []

    cv2.waitKey(1)