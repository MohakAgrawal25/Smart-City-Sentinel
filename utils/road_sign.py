import cv2
import numpy as np

def get_condition(crop):
    crop = cv2.resize(crop, (128,128))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    avg_saturation = hsv[:,:,1].mean()
    avg_brightness = hsv[:,:,2].mean()

    edges = cv2.Canny(gray, 50 if avg_brightness > 100 else 100, 200)
    edge_density = np.sum(edges>0)/(edges.shape[0]*edges.shape[1])

    penalty = 0
    if edge_density > 0.10: penalty += 2
    if lap_var > 1000: penalty += 1
    if avg_saturation < 35: penalty += 1
    if avg_brightness < 60 and lap_var > 800:
        penalty -= 1

    return "Poor" if penalty >= 2 else "Good"