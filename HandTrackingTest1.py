import cv2
import mediapipe as mp
import numpy as np
import time

#640x480
#1280x720

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
#cap.set(10,100)

print(cap.get(3), cap.get(4))

SCREEN_WIDTH = int(cap.get(3))
SCREEN_HEIGHT = int(cap.get(4))

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

handCenterX = 0
handCenterY = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    bgImg = np.zeros((SCREEN_HEIGHT,SCREEN_WIDTH,3), np.uint8)

    shownImg = bgImg

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = shownImg.shape
                cx, cy = int(lm.x*w), int(lm.y*h)

                handCenterX = cx
                handCenterY = cy

                if id == 9:
                    if handCenterX < (SCREEN_WIDTH/3):
                        print("LEFT")
                    elif handCenterX < (SCREEN_WIDTH*2/3):
                        print("CENTER")
                    else:
                        print("RIGHT")

                    #cv2.circle(shownImg, (cx, cy), 15, (255,0,0), cv2.FILLED)

            mpDraw.draw_landmarks(shownImg, handLms, mpHands.HAND_CONNECTIONS)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(shownImg, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image BlackBG", shownImg)
    cv2.imshow("Image", img)

    handCenterX = 0
    handCenterY = 0

    cv2.waitKey(1)