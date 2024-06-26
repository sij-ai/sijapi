import requests
import os
import json

filename = 'location_log.json'
server = 'https://api.sij.ai'

def upload_location_data(data):
    headers = {
        'Authorization': 'Bearer sk-NhrtQwCHNdK5sRZC',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(f'{server}/locate', json=data, headers=headers)
        if response.status_code == 200:
            print('Location and weather updated successfully.')
            os.remove(filename)
        else:
            print(f'Failed to post data. Status code: {response.status_code}')
            print(response.text)
    except requests.RequestException as e:
        print(f'Error posting data: {e}')

if not os.path.exists(filename):
    print('No data to upload.')
else:
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        # Ensure all datetime fields are correctly named and add default context if missing
        for location in data:
            if 'date' in location:
                location['datetime'] = location.pop('date')
            # Ensure context dictionary exists with all required keys
            if 'context' not in location:
                location['context'] = {
                    'action': 'manual',
                    'device_type': 'Pythonista',
                    'device_model': None,
                    'device_name': None,
                    'device_os': None
                }
            else:
                context = location['context']
                context.setdefault('action', 'manual')
                context.setdefault('device_type', 'Pythonista')
                context.setdefault('device_model', None)
                context.setdefault('device_name', None)
                context.setdefault('device_os', None)
        upload_location_data(data)
    except FileNotFoundError:
        print(f'File {filename} not found.')
    except json.JSONDecodeError:
        print(f'Error decoding JSON from {filename}.')
    except Exception as e:
        print(f'Unexpected error: {e}')
