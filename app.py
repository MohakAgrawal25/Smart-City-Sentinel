import os
import json
from flask import Flask, render_template, request
from utils.detector import run_detection

app = Flask(__name__)

os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/outputs", exist_ok=True)

MARKERS_FILE = "markers.json"

def load_markers():
    if os.path.exists(MARKERS_FILE):
        with open(MARKERS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_markers(markers):
    with open(MARKERS_FILE, 'w') as f:
        json.dump(markers, f, indent=2)

all_markers = load_markers()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global all_markers
    file = request.files.get("image")
    if not file:
        return "No file uploaded", 400

    path = os.path.join("static/uploads", file.filename)
    file.save(path)

    outputs, new_markers = run_detection(path, file.filename)

    if new_markers:
        all_markers.extend(new_markers)
        save_markers(all_markers)
        print(f"✅ Added {len(new_markers)} markers. Total: {len(all_markers)}")

    return render_template("result.html", outputs=outputs, markers=new_markers)

@app.route("/map")
def map_view():
    markers = load_markers()
    print(f"🗺️ Rendering map with {len(markers)} markers")
    return render_template("map.html", markers=markers)

if __name__ == "__main__":
    app.run(debug=True)