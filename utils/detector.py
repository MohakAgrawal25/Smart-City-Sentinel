import cv2
import os
import numpy as np
from ultralytics import YOLO
from utils.road_sign import get_condition
from utils.location import extract_lat_long

# Load models
models = {
    "pothole_cracks": YOLO("models/pothole_cracks.pt"),
    "garbage": YOLO("models/garbage_det.pt"),
    "fallen_tree": YOLO("models/Tree.pt"),
    "electric_pole": YOLO("models/Pole (1).pt"),
    "street_light": YOLO("models/street_light_det_og.pt"),
    "road_sign": YOLO("models/best (2).pt")
}

# Colors (BGR)
colors = {
    "pothole_cracks": (0, 0, 255),
    "garbage": (255, 0, 0),
    "fallen_tree": (0, 255, 0),
    "electric_pole": (128, 0, 128),
    "street_light": (0, 255, 255),
    "road_sign": (0, 165, 255)
}

# Drawing parameters
BOX_THICKNESS = 6
FONT_SCALE = 1.2
FONT_THICKNESS = 3
TEXT_OFFSET = 15
MASK_ALPHA = 0.5

def draw_mask(img, mask_points, color):
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(mask_points, dtype=np.int32)], color)
    cv2.addWeighted(overlay, MASK_ALPHA, img, 1 - MASK_ALPHA, 0, img)

def extract_condition(value):
    """Extract condition (e.g., 'Poor', 'Good') from get_condition return."""
    s = str(value).upper()
    if "POOR" in s:
        return "POOR"
    elif "GOOD" in s:
        return "GOOD"
    if isinstance(value, (tuple, list)) and len(value) > 0:
        return str(value[0])
    return str(value)

def run_detection(img_path, filename):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Could not read image: {img_path}")
        return {}, []

    lat, lon = extract_lat_long(filename)
    base = os.path.splitext(filename)[0]
    img_rel_path = f"uploads/{filename}"

    outputs = {}
    markers = []

    for name, model in models.items():
        results = model(img, verbose=False)
        img_copy = img.copy()
        has_masks = results[0].masks is not None
        any_detection = False

        for i, box in enumerate(results[0].boxes):
            conf = box.conf.item()
            if conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = name
            is_issue = True
            color = colors.get(name, (0, 0, 255))

            # --- Street Light (using your Jupyter notebook logic) ---
            if name == "street_light":
                # Crop the top part of the bounding box (where the bulb is)
                bulb_y2 = y1 + int((y2 - y1) * 0.25)
                crop = img[y1:bulb_y2, x1:x2]

                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray)

                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    mask_white  = cv2.inRange(hsv, (0,   0,   200), (180, 60,  255))
                    mask_yellow = cv2.inRange(hsv, (15,  80,  150), (40,  255, 255))
                    glow_pixels = cv2.countNonZero(mask_white) + cv2.countNonZero(mask_yellow)

                    is_on = brightness > 80 or glow_pixels > 30
                else:
                    is_on = False

                status = "ON" if is_on else "OFF"
                label = f"Light {status}"
                if status == "ON":
                    is_issue = False
                    color = (0, 255, 0)  # green box
                else:
                    is_issue = True      # OFF → marker
                print(f"Street light: {status} (brightness: {brightness:.1f}, glow: {glow_pixels}) → is_issue={is_issue}")

            # --- Road Sign ---
            elif name == "road_sign":
                crop = img[y1:y2, x1:x2]
                raw_cond = get_condition(crop) if crop.size > 0 else "Unknown"
                condition = extract_condition(raw_cond)
                label = f"Sign {condition}"
                if condition.upper() == "POOR":
                    is_issue = True
                    color = colors.get(name, (0, 165, 255))
                else:
                    is_issue = False
                    color = (0, 255, 0)  # green box
                print(f"Road sign: {condition} → is_issue={is_issue}")

            # Append confidence percentage
            conf_percent = int(conf * 100)
            label = f"{label} ({conf_percent}%)"

            # Add marker only if it's an issue
            if is_issue:
                if lat is not None and lon is not None:
                    markers.append([name, lat, lon, img_rel_path])

            # Draw on image (always)
            any_detection = True
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            if has_masks:
                mask_xy = results[0].masks.xy[i]
                if len(mask_xy) > 0:
                    mask_points = mask_xy.reshape(-1, 2).astype(np.int32)
                    draw_mask(img_copy, mask_points, color)

            # Put label above box
            label_y = int(y1 - TEXT_OFFSET) if y1 - TEXT_OFFSET > TEXT_OFFSET else y1 + TEXT_OFFSET
            cv2.putText(img_copy, label, (int(x1), label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)

        # Save output image if any detection occurred for this model
        if any_detection:
            out_filename = f"outputs/{base}_{name}.jpg"
            out_path = os.path.join("static", out_filename)
            success = cv2.imwrite(out_path, img_copy)
            if success:
                outputs[name] = out_filename
                print(f"✅ Saved: {out_path}")
            else:
                print(f"❌ Failed to save: {out_path}")
        else:
            print(f"ℹ️ No detection for {name} in {filename}")

    # Hybrid marker if multiple issue types at same location
    unique_types = set([m[0] for m in markers])
    if len(unique_types) > 1 and lat is not None and lon is not None:
        first_img = markers[0][3] if markers else img_rel_path
        markers.append(["hybrid", lat, lon, first_img])

    return outputs, markers