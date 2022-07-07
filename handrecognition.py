import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

wCam = int(cap.get(3))
hCam = int(cap.get(4))

avg_frames = 5 
prevGestures = [] # gestures calculated in previous frames

# Getting media-pipe ready
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=.7,max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Vars used to calculate avg fps
prevTime = 0
currTime = 0
fpsList = []


def normalize(v):
    mag = np.sqrt(v[0] ** 2 + v[1] ** 2)
    v[0] = v[0] / mag
    v[1] = v[1] / mag
    return v

def gesture(f):
    """
    Uses the open fingers list to recognize gestures
    :param f: list of open fingers (+) and closed fingers (-)
    :return: string representing the gesture that is detected
    """
    #print("Thumb is at:", f[0])
    if f[1] > 0 > f[2] and f[4] > 0 > f[3]:
        return "Rock & Roll"
    elif f[0] > 0 and (f[1] < 0 and f[2] < 0 and f[3] < 0 and f[4] < 0):
        return "Thumbs Up"
    elif f[0] < 0 and f[1] > 0 and f[2] < 0 and (f[3] < 0 and f[4] < 0):
        return "1 finger"
    elif f[0] < 0 and f[1] > 0 and f[2] > 0 and (f[3] < 0 and f[4] < 0):
        return "Peace/2 fingers"
    elif f[0] > 0 and f[1] > 0 and f[2] > 0 and f[3] > 0 and f[4] > 0:
        return "Open Hand"
    elif f[0] < 0 and f[1] < 0 and f[2] < 0 and f[3] < 0 and f[4] < 0:
        return "Fist"
    elif f[0] < 0 and f[1] > 0 and f[2] > 0 and f[3] > 0 and f[4] > 0: 
        return "4 fingers"
    elif f[0] < 0 and f[1] > 0 and f[2] > 0 and f[3] > 0 and f[4] < 0:
        return "3 fingers"
    elif f[0] < 0 and f[1] < 0 and f[2] > 0 and f[3] < 0 and f[4] < 0:
        return "Big OOOOF"
    else:
        return "No Gesture"



def straightFingers(hand, img):
    """
    Calculates which fingers are open and which fingers are closed
    :param hand: media-pipe object of the hand
    :param img: frame with the hand in it
    :return: list of open (+) and closed (-) fingers
    """
    TipIDs = [4, 8, 12, 16, 20]  # list of the id's for the finger tip landmarks
    openFingers = []
    
    lms = hand.landmark  # 2d list of all 21 landmarks with there respective x, an y coordinates
    #print(lms)
    for id in TipIDs:
        if id == 4: # This is for the thumb calculation, because it works differently than the other fingers
            x2, y2 = lms[id].x, lms[id].y  # x, and y of the finger tip
            x1, y1 = lms[id-2].x, lms[id-2].y  # x, and y of the joint 2 points below the finger tip
            x0, y0 = lms[0].x, lms[0].y  # x, and y of the wrist
            fv = [x2-x1, y2-y1]  # joint to finger tip vector
            fv = normalize(fv)
            pv = [x1-x0, y1-y0]  # wrist to joint vector
            pv = normalize(pv)

            thumb = np.dot(fv,pv)

            if thumb > .65:
                openFingers.append(thumb)  # Calculates if the finger is open or closed
            else:
                openFingers.append(-1)

        else: # for any other finger (not thumb)
            tipx, tipy = lms[id].x, lms[id].y  # x, and y of the finger tip
            x1, y1 = lms[id-2].x, lms[id-2].y  # x, and y of the joint 2 points below the finger tip
            x0, y0 = lms[0].x, lms[0].y  # x, and y of the wrist
            fv = [tipx-x1, tipy-y1]  # joint to finger tip vector
            fv = normalize(fv)
            pv = [x1-x0, y1-y0]  # wrist to joint vector
            pv = normalize(pv)
            openFingers.append(np.dot(fv,pv))  # Calculates if the finger is open or closed

    return openFingers

frame_count = 0

while True:
    if(cv2.waitKey(1))== 27: #HIT ESCAPE TO CLOSE
        break

    #Gets the image from openCV
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #calculate open fingers if hand is in frame
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            fingers = straightFingers(handLms, img)
            prevGestures.append(gesture(fingers))
            frame_count += 1
            mpDraw.draw_landmarks(img, handLms)
        if frame_count > (avg_frames - 1):
            if (all(x == prevGestures[0] for x in prevGestures)):
                print(prevGestures[0])
                prevGestures.remove(prevGestures[0])
            prevGestures = []
            frame_count = 0

   

    cv2.imshow("Video with Hand Detection (Esc to leave)", img)

    



