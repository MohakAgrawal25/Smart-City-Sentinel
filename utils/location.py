import os

def extract_lat_long(filename):
    try:
        name = os.path.splitext(filename)[0]
        coords = name.split('_')[-1]
        if ',' in coords:
            lat, lon = coords.split(',')
            return float(lat), float(lon)
    except Exception as e:
        print(f"❌ Error parsing coordinates from {filename}: {e}")
    return None, None
