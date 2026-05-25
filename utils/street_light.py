import cv2
import numpy as np

def check_street_light(img, boxes):
    results = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        bulb_y2 = y1 + int((y2 - y1) * 0.25)
        crop = img[y1:bulb_y2, x1:x2]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask_white  = cv2.inRange(hsv, (0,0,200),(180,60,255))
        mask_yellow = cv2.inRange(hsv, (15,80,150),(40,255,255))
        glow_pixels = cv2.countNonZero(mask_white) + cv2.countNonZero(mask_yellow)

        is_on = brightness > 80 or glow_pixels > 30
        results.append(("ON" if is_on else "OFF", (x1,y1,x2,y2)))

    return results