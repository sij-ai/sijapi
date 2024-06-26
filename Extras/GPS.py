import location
import time
import json
import os
import sys
from datetime import datetime, timezone

def get_current_location():
    location.start_updates()
    time.sleep(1)  # Give it a moment to get an accurate fix
    current_location = location.get_location()
    location.stop_updates()

    elevation = current_location['altitude']
    latitude = current_location['latitude']
    longitude = current_location['longitude']
    current_time = datetime.now(timezone.utc)
    timestamp = current_time.isoformat()

    return {
        'latitude': latitude,
        'longitude': longitude,
        'elevation': elevation,
        'datetime': timestamp
    }

def save_location_data(data, context, filename='location_log.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    data['context'] = context
    existing_data.append(data)

    with open(filename, 'w') as f:
        json.dump(existing_data, f, indent=4)

if len(sys.argv) > 1:
    try:
        context = json.loads(sys.argv[1])
        context.setdefault('action', 'manual')
        context.setdefault('device_type', 'Pythonista')
        context.setdefault('device_model', None)
        context.setdefault('device_name', None)
        context.setdefault('device_os', None)
    except json.JSONDecodeError:
        print("Error: The provided argument is not a valid JSON.")
        sys.exit(1)
else:
    context = {
        'action': 'manual',
        'device_type': 'Pythonista',
        'device_model': None,
        'device_name': None,
        'device_os': None
    }

location_data = get_current_location()
save_location_data(location_data, context)
print(f"Location data: {location_data} with context '{context}' saved locally.")
time.sleep(5)
