import cv2
import mediapipe as mp
import numpy as np
import time

#640x480
#1280x720
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

cap = cv2.VideoCapture(0)
cap.set(3,SCREEN_WIDTH)
cap.set(4,SCREEN_HEIGHT)
cap.set(10, 100)
print(cap.get(4))

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

lmX = []
lmY = []

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    bgImg = np.zeros((SCREEN_HEIGHT,SCREEN_WIDTH,3), np.uint8)
    #bgImg = np.zeros((1650,3000,3), np.uint8)

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

                #if id == 8:
                #    cv2.circle(img, (cx, cy), 15, (255,0,0),cv2.FILLED)
            #print(handLms.landmark) 5,6,7,8,

            mpDraw.draw_landmarks(shownImg, handLms, mpHands.HAND_CONNECTIONS)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(shownImg, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image BlackBG", shownImg)
    cv2.imshow("Image", img)

    lmX = []
    lmY = []

    cv2.waitKey(1)