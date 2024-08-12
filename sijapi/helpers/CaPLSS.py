#!/usr/bin/env python3

import requests
import json
import time
import os
import subprocess
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime

# Environment variables for database connection
DB_NAME = os.getenv('DB_NAME', 'sij')
DB_USER = os.getenv('DB_USER', 'sij')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'Synchr0!')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')

def get_feature_count(url):
    params = {
        'where': '1=1',
        'returnCountOnly': 'true',
        'f': 'json'
    }
    retries = Retry(total=10, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retries))
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    return data.get('count', 0)

def fetch_features(url, offset, num, max_retries=5):
    params = {
        'where': '1=1',
        'outFields': '*',
        'geometryPrecision': 6,
        'outSR': 4326,
        'f': 'json',
        'resultOffset': offset,
        'resultRecordCount': num
    }
    for attempt in range(max_retries):
        try:
            retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
            with requests.Session() as session:
                session.mount("https://", HTTPAdapter(max_retries=retries))
                response = session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching features (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))  # Exponential backoff


def download_layer(layer_num, layer_name):
    base_dir = os.path.expanduser('~/data')
    os.makedirs(base_dir, exist_ok=True)

    file_path = os.path.join(base_dir, f'PLSS_{layer_name}.geojson')
    temp_file_path = os.path.join(base_dir, f'PLSS_{layer_name}_temp.json')

    url = f"https://gis.blm.gov/arcgis/rest/services/Cadastral/BLM_Natl_PLSS_CadNSDI/MapServer/{layer_num}/query"

    total_count = get_feature_count(url)
    print(f"Total {layer_name} features: {total_count}")

    batch_size = 1000
    chunk_size = 10000  # Write to file every 10,000 features
    offset = 0
    all_features = []

    # Check if temporary file exists and load its content
    if os.path.exists(temp_file_path):
        try:
            with open(temp_file_path, 'r') as f:
                all_features = json.load(f)
            offset = len(all_features)
            print(f"Resuming download from offset {offset}")
        except json.JSONDecodeError:
            print("Error reading temporary file. Starting download from the beginning.")
            offset = 0
            all_features = []

    try:
        while offset < total_count:
            print(f"Fetching {layer_name} features {offset} to {offset + batch_size}...")
            data = fetch_features(url, offset, batch_size)

            new_features = data.get('features', [])
            if not new_features:
                break

            all_features.extend(new_features)
            offset += len(new_features)

            # Progress indicator
            progress = offset / total_count
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            print(f'\rProgress: [{bar}] {progress:.1%} ({offset}/{total_count} features)', end='', flush=True)

            # Save progress to temporary file every chunk_size features
            if len(all_features) % chunk_size == 0:
                with open(temp_file_path, 'w') as f:
                    json.dump(all_features, f)

            time.sleep(1)

        print(f"\nTotal {layer_name} features fetched: {len(all_features)}")

        # Write final GeoJSON file
        with open(file_path, 'w') as f:
            f.write('{"type": "FeatureCollection", "features": [\n')
            for i, feature in enumerate(all_features):
                geojson_feature = {
                    "type": "Feature",
                    "properties": feature['attributes'],
                    "geometry": feature['geometry']
                }
                json.dump(geojson_feature, f)
                if i < len(all_features) - 1:
                    f.write(',\n')
            f.write('\n]}')

        print(f"GeoJSON file saved as '{file_path}'")

        # Remove temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return file_path
    except Exception as e:
        print(f"\nError during download: {e}")
        print(f"Partial data saved in {temp_file_path}")
        return None


def check_postgres_connection():
    try:
        subprocess.run(['psql', '-h', DB_HOST, '-p', DB_PORT, '-U', DB_USER, '-d', DB_NAME, '-c', 'SELECT 1;'],
                       check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False

def check_postgis_extension():
    try:
        result = subprocess.run(['psql', '-h', DB_HOST, '-p', DB_PORT, '-U', DB_USER, '-d', DB_NAME,
                                 '-c', "SELECT 1 FROM pg_extension WHERE extname = 'postgis';"],
                                check=True, capture_output=True, text=True)
        return '1' in result.stdout
    except subprocess.CalledProcessError:
        return False

def create_postgis_extension():
    try:
        subprocess.run(['psql', '-h', DB_HOST, '-p', DB_PORT, '-U', DB_USER, '-d', DB_NAME,
                        '-c', "CREATE EXTENSION IF NOT EXISTS postgis;"],
                       check=True, capture_output=True, text=True)
        print("PostGIS extension created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating PostGIS extension: {e}")
        sys.exit(1)

def import_to_postgis(file_path, table_name):
    if not check_postgres_connection():
        print("Error: Unable to connect to PostgreSQL. Please check your connection settings.")
        sys.exit(1)

    if not check_postgis_extension():
        print("PostGIS extension not found. Attempting to create it...")
        create_postgis_extension()

    ogr2ogr_command = [
        'ogr2ogr',
        '-f', 'PostgreSQL',
        f'PG:dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}',
        file_path,
        '-nln', table_name,
        '-overwrite'
    ]

    try:
        subprocess.run(ogr2ogr_command, check=True, capture_output=True, text=True)
        print(f"Data successfully imported into PostGIS table: {table_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error importing data into PostGIS: {e}")
        print(f"Command that failed: {e.cmd}")
        print(f"Error output: {e.stderr}")

def check_ogr2ogr():
    try:
        subprocess.run(['ogr2ogr', '--version'], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False


def main():
    if not check_ogr2ogr():
        print("Error: ogr2ogr not found. Please install GDAL/OGR tools.")
        print("On Debian: sudo apt-get install gdal-bin")
        print("On macOS with Homebrew: brew install gdal")
        sys.exit(1)

    try:
        township_file = os.path.expanduser('~/data/PLSS_Townships.geojson')
        if not os.path.exists(township_file):
            township_file = download_layer(1, "Townships")
        if township_file:
            import_to_postgis(township_file, "public.plss_townships")
        else:
            print("Failed to download Townships data. Skipping import.")

        section_file = os.path.expanduser('~/data/PLSS_Sections.geojson')
        if not os.path.exists(section_file):
            section_file = download_layer(2, "Sections")
        if section_file:
            import_to_postgis(section_file, "public.plss_sections")
        else:
            print("Failed to download Sections data. Skipping import.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
