# CaPLSS_downloader_and_importer.py
import requests
import json
import time
import os
import subprocess
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_feature_count(url):
    params = {
        'where': '1=1',
        'returnCountOnly': 'true',
        'f': 'json'
    }
    retries = Retry(total=10, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retries)
    session = requests.Session()
    session.mount("https://", adapter)

    response = session.get(url, params=params, timeout=15)  # Add timeout parameter
    response.raise_for_status()
    data = response.json()
    return data.get('count', 0)


def fetch_features(url, offset, num):
    params = {
        'where': '1=1',
        'outFields': '*',
        'geometryPrecision': 6,
        'outSR': 4326,
        'f': 'json',
        'resultOffset': offset,
        'resultRecordCount': num
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def download_layer(layer_num, layer_name):
    url = f"https://gis.blm.gov/arcgis/rest/services/Cadastral/BLM_Natl_PLSS_CadNSDI/MapServer/{layer_num}/query"
    
    total_count = get_feature_count(url)
    print(f"Total {layer_name} features: {total_count}")

    batch_size = 1000
    offset = 0
    all_features = []

    while offset < total_count:
        print(f"Fetching {layer_name} features {offset} to {offset + batch_size}...")
        data = fetch_features(url, offset, batch_size)
        
        new_features = data.get('features', [])
        if not new_features:
            break

        all_features.extend(new_features)
        offset += len(new_features)

        print(f"Progress: {len(all_features)}/{total_count} features")

        time.sleep(1)  # Be nice to the server

    print(f"Total {layer_name} features fetched: {len(all_features)}")

    # Convert to GeoJSON
    geojson_features = [
        {
            "type": "Feature",
            "properties": feature['attributes'],
            "geometry": feature['geometry']
        } for feature in all_features
    ]

    full_geojson = {
        "type": "FeatureCollection",
        "features": geojson_features
    }

    # Define a base directory that exists on both macOS and Debian
    base_dir = os.path.expanduser('~/data')
    os.makedirs(base_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    # Use os.path.join to construct the file path
    file_path = os.path.join(base_dir, f'PLSS_{layer_name}.geojson')
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(full_geojson, f)
    
    print(f"GeoJSON file saved as '{file_path}'")
    
    return file_path

def import_to_postgis(file_path, table_name):
    db_name = 'sij'
    db_user = 'sij'
    db_password = 'Synchr0!'

    ogr2ogr_command = [
        'ogr2ogr',
        '-f', 'PostgreSQL',
        f'PG:dbname={db_name} user={db_user} password={db_password}',
        file_path,
        '-nln', table_name,
        '-overwrite'
    ]

    subprocess.run(ogr2ogr_command, check=True)
    print(f"Data successfully imported into PostGIS table: {table_name}")

def main():
    try:
        # Download and import Townships (Layer 1)
        township_file = download_layer(1, "Townships")
        import_to_postgis(township_file, "public.plss_townships")

        # Download and import Sections (Layer 2)
        section_file = download_layer(2, "Sections")
        import_to_postgis(section_file, "public.plss_sections")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error importing data into PostGIS: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
