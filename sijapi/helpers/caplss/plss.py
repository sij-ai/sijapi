#!/usr/bin/env python3

import requests
import json
import time
import os
import subprocess
import sys
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import argparse
import psycopg2
from psycopg2.extras import execute_values

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys_config_path = os.path.join(script_dir, '..', 'config', 'sys.yaml')
    gis_config_path = os.path.join(script_dir, '..', 'config', 'gis.yaml')

    with open(sys_config_path, 'r') as f:
        sys_config = yaml.safe_load(f)

    with open(gis_config_path, 'r') as f:
        gis_config = yaml.safe_load(f)

    return sys_config, gis_config

def get_db_config(sys_config):
    pool = sys_config.get('POOL', [])
    if pool:
        db_config = pool[0]
        return {
            'DB_NAME': db_config.get('db_name'),
            'DB_USER': db_config.get('db_user'),
            'DB_PASSWORD': db_config.get('db_pass'),
            'DB_HOST': db_config.get('ts_ip'),
            'DB_PORT': str(db_config.get('db_port'))
        }
    return {}

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
        'resultRecordCount': num,
        'orderByFields': 'OBJECTID'
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


def create_table(db_config, table_name, gis_config):
    conn = psycopg2.connect(
        dbname=db_config['DB_NAME'],
        user=db_config['DB_USER'],
        password=db_config['DB_PASSWORD'],
        host=db_config['DB_HOST'],
        port=db_config['DB_PORT']
    )
    try:
        with conn.cursor() as cur:
            # Check if the table already exists
            cur.execute(f"SELECT to_regclass('{table_name}')")
            if cur.fetchone()[0] is None:
                # If the table doesn't exist, create it based on the first feature
                url = next(layer['url'] for layer in gis_config['layers'] if layer['table_name'] == table_name)
                first_feature = fetch_features(url, 0, 1)['features'][0]
                columns = []
                for attr, value in first_feature['attributes'].items():
                    column_name = attr.lower().replace('.', '_').replace('()', '')
                    if isinstance(value, int):
                        columns.append(f'"{column_name}" INTEGER')
                    elif isinstance(value, float):
                        columns.append(f'"{column_name}" DOUBLE PRECISION')
                    else:
                        columns.append(f'"{column_name}" TEXT')
                
                create_sql = f"""
                CREATE TABLE {table_name} (
                    id SERIAL PRIMARY KEY,
                    geom GEOMETRY(Polygon, 4326),
                    {', '.join(columns)}
                )
                """
                cur.execute(create_sql)
                
                # Create index on plssid
                cur.execute(f'CREATE INDEX idx_{table_name.split(".")[-1]}_plssid ON {table_name}("plssid")')
                
                print(f"Created table: {table_name}")
            else:
                print(f"Table {table_name} already exists")
        conn.commit()
    except psycopg2.Error as e:
        print(f"Error creating table {table_name}: {e}")
    finally:
        conn.close()


def insert_features_to_db(features, table_name, db_config):
    conn = psycopg2.connect(
        dbname=db_config['DB_NAME'],
        user=db_config['DB_USER'],
        password=db_config['DB_PASSWORD'],
        host=db_config['DB_HOST'],
        port=db_config['DB_PORT']
    )
    try:
        with conn.cursor() as cur:
            # Get the column names from the table
            cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name.split('.')[-1]}'")
            db_columns = [row[0] for row in cur.fetchall() if row[0] != 'id']
            
            # Prepare the SQL statement
            sql = f"""
            INSERT INTO {table_name} ({', '.join([f'"{col}"' for col in db_columns])})
            VALUES %s
            """
            
            # Prepare the template for execute_values
            template = f"({', '.join(['%s' for _ in db_columns])})"
            
            values = []
            for feature in features:
                geom = feature.get('geometry')
                attrs = feature.get('attributes')
                if geom and attrs:
                    rings = geom['rings'][0]
                    wkt = f"POLYGON(({','.join([f'{x} {y}' for x, y in rings])}))"
                    
                    row = []
                    for col in db_columns:
                        if col == 'geom':
                            row.append(wkt)
                        else:
                            # Map database column names back to original attribute names
                            attr_name = col.upper()
                            if attr_name == 'SHAPE_STAREA':
                                attr_name = 'Shape.STArea()'
                            elif attr_name == 'SHAPE_STLENGTH':
                                attr_name = 'Shape.STLength()'
                            row.append(attrs.get(attr_name))
                    
                    values.append(tuple(row))
                else:
                    print(f"Skipping invalid feature: {feature}")
            
            if values:
                execute_values(cur, sql, values, template=template, page_size=100)
                print(f"Inserted {len(values)} features")
            else:
                print("No valid features to insert")
        conn.commit()
    except Exception as e:
        print(f"Error inserting features: {e}")
        print(f"First feature for debugging: {features[0] if features else 'No features'}")
        conn.rollback()
    finally:
        conn.close()



def download_and_import_layer(layer_config, db_config, gis_config, force_refresh):
    url = layer_config['url']
    layer_name = layer_config['layer_name']
    table_name = layer_config['table_name']
    batch_size = layer_config['batch_size']
    delay = layer_config['delay'] / 1000  # Convert to seconds

    total_count = get_feature_count(url)
    print(f"Total {layer_name} features: {total_count}")

    # Check existing records in the database
    existing_count = get_existing_record_count(db_config, table_name)

    if existing_count == total_count and not force_refresh:
        print(f"Table {table_name} already contains all {total_count} features. Skipping.")
        return

    if force_refresh:
        delete_existing_table(db_config, table_name)
        create_table(db_config, table_name, gis_config)
        existing_count = 0
    elif existing_count == 0:
        create_table(db_config, table_name, gis_config)

    offset = existing_count

    start_time = time.time()
    try:
        while offset < total_count:
            batch_start_time = time.time()
            print(f"Fetching {layer_name} features {offset} to {offset + batch_size}...")
            try:
                data = fetch_features(url, offset, batch_size)
                new_features = data.get('features', [])
                if not new_features:
                    break

                insert_features_to_db(new_features, table_name, db_config)
                offset += len(new_features)

                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                print(f"Batch processed in {batch_duration:.2f} seconds")

                # Progress indicator
                progress = offset / total_count
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                print(f'\rProgress: [{bar}] {progress:.1%} ({offset}/{total_count} features)', end='', flush=True)

                time.sleep(delay)
            except Exception as e:
                print(f"\nError processing batch starting at offset {offset}: {e}")
                print("Continuing with next batch...")
                offset += batch_size

        end_time = time.time()
        total_duration = end_time - start_time
        print(f"\nTotal {layer_name} features fetched and imported: {offset}")
        print(f"Total time: {total_duration:.2f} seconds")

    except Exception as e:
        print(f"\nError during download and import: {e}")
        print(f"Last successful offset: {offset}")

def get_existing_record_count(db_config, table_name):
    conn = psycopg2.connect(
        dbname=db_config['DB_NAME'],
        user=db_config['DB_USER'],
        password=db_config['DB_PASSWORD'],
        host=db_config['DB_HOST'],
        port=db_config['DB_PORT']
    )
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()[0]
        return count
    except psycopg2.Error:
        return 0
    finally:
        conn.close()

def delete_existing_table(db_config, table_name):
    conn = psycopg2.connect(
        dbname=db_config['DB_NAME'],
        user=db_config['DB_USER'],
        password=db_config['DB_PASSWORD'],
        host=db_config['DB_HOST'],
        port=db_config['DB_PORT']
    )
    try:
        with conn.cursor() as cur:
            # Drop the index if it exists
            cur.execute(f"DROP INDEX IF EXISTS idx_{table_name.split('.')[-1]}_plssid")
            
            # Then drop the table
            cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
        conn.commit()
        print(f"Deleted existing table and index: {table_name}")
    except psycopg2.Error as e:
        print(f"Error deleting table {table_name}: {e}")
    finally:
        conn.close()


def check_postgres_connection(db_config):
    try:
        subprocess.run(['psql',
                        '-h', db_config['DB_HOST'],
                        '-p', db_config['DB_PORT'],
                        '-U', db_config['DB_USER'],
                        '-d', db_config['DB_NAME'],
                        '-c', 'SELECT 1;'],
                       check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False

def check_postgis_extension(db_config):
    try:
        result = subprocess.run(['psql',
                                 '-h', db_config['DB_HOST'],
                                 '-p', db_config['DB_PORT'],
                                 '-U', db_config['DB_USER'],
                                 '-d', db_config['DB_NAME'],
                                 '-c', "SELECT 1 FROM pg_extension WHERE extname = 'postgis';"],
                                check=True, capture_output=True, text=True)
        return '1' in result.stdout
    except subprocess.CalledProcessError:
        return False

def create_postgis_extension(db_config):
    try:
        subprocess.run(['psql',
                        '-h', db_config['DB_HOST'],
                        '-p', db_config['DB_PORT'],
                        '-U', db_config['DB_USER'],
                        '-d', db_config['DB_NAME'],
                        '-c', "CREATE EXTENSION IF NOT EXISTS postgis;"],
                       check=True, capture_output=True, text=True)
        print("PostGIS extension created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating PostGIS extension: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download and import PLSS data")
    parser.add_argument("--force-refresh", nargs='*', help="Force refresh of specified layers or all if none specified")
    args = parser.parse_args()

    sys_config, gis_config = load_config()
    db_config = get_db_config(sys_config)

    if not check_postgres_connection(db_config):
        print("Error: Unable to connect to PostgreSQL. Please check your connection settings.")
        sys.exit(1)

    if not check_postgis_extension(db_config):
        print("PostGIS extension not found. Attempting to create it...")
        create_postgis_extension(db_config)

    try:
        for layer in gis_config['layers']:
            if args.force_refresh is None or not args.force_refresh or layer['layer_name'] in args.force_refresh:
                download_and_import_layer(layer, db_config, gis_config, bool(args.force_refresh))
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
