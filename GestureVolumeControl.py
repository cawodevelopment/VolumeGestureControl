import cv2 as cv
import numpy as np
import math
import HandTrackingModule as htm
from pycaw.pycaw import AudioUtilities

capture = cv.VideoCapture(0)

detector = htm.HandDetector(min_detection_confidence=0.7)

device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume

volumeRange = volume.GetVolumeRange()

minVol = volumeRange[0]
maxVol = volumeRange[1]


while True:
    ret, frame = capture.read()
    flipped_frame = cv.flip(frame, 1)

    find_hands = detector.findHands(flipped_frame)
    lmList = detector.findPosition(find_hands, draw=False)

    if len(lmList) != 0:

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv.circle(flipped_frame, (lmList[4][1], lmList[4][2]), 15, (255, 0, 255), cv.FILLED)
        cv.circle(flipped_frame, (lmList[8][1], lmList[8][2]), 15, (255, 0, 255), cv.FILLED)
        cv.line(flipped_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2-x1, y2-y1)

        vol = np.interp(length, [40, 250], [minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)

    cv.imshow("Webcam", flipped_frame)

    if cv.waitKey(20) & 0xFF == ord("d"):
        break

capture.release()
cv.destroyAllWindows()