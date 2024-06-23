#!/bin/bash

DB_NAME="weatherlocate.db"

# Step 1: Backup existing data
echo "Backing up existing data..."
sqlite3 $DB_NAME <<EOF
.headers on
.mode csv
.output hourly_weather_backup.csv
SELECT * FROM HourlyWeather;
.output daily_weather_backup.csv
SELECT * FROM DailyWeather;
.output hours_backup.csv
SELECT * FROM Hours;
.output days_backup.csv
SELECT * FROM Days;
EOF

# Step 2: Drop and recreate tables
echo "Dropping and recreating tables..."
sqlite3 $DB_NAME <<EOF
DROP TABLE IF EXISTS HourlyWeather;
DROP TABLE IF EXISTS DailyWeather;
DROP TABLE IF EXISTS Hours;
DROP TABLE IF EXISTS Days;

CREATE TABLE HourlyWeather (
    id INTEGER PRIMARY KEY,
    datetime TEXT NOT NULL,
    temp REAL,
    feelslike REAL,
    humidity REAL,
    dew REAL,
    precip REAL,
    precipprob REAL,
    snow REAL,
    snowdepth REAL,
    windgust REAL,
    windspeed REAL,
    winddir REAL,
    pressure REAL,
    cloudcover REAL,
    visibility REAL,
    solarradiation REAL,
    solarenergy REAL,
    uvindex REAL,
    severerisk REAL,
    conditions TEXT,
    icon TEXT,
    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE DailyWeather (
    id INTEGER PRIMARY KEY,
    sunrise_time TEXT,
    sunset_time TEXT,
    description TEXT,
    tempmax REAL,
    tempmin REAL,
    uvindex REAL,
    winddir REAL,
    windspeedmean REAL,
    windspeed REAL,
    icon TEXT,
    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Hours (
    id INTEGER PRIMARY KEY,
    day_id INTEGER,
    hour INTEGER,
    hourly_weather_id INTEGER,
    FOREIGN KEY (day_id) REFERENCES Days(id),
    FOREIGN KEY (hourly_weather_id) REFERENCES HourlyWeather(id)
);

CREATE TABLE Days (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    daily_weather_id INTEGER,
    FOREIGN KEY (daily_weather_id) REFERENCES DailyWeather(id)
);
EOF

# Step 3: Import data from backup files
echo "Importing data from backup files..."

python3 <<EOF
import sqlite3
import csv
from datetime import datetime

def import_data():
    conn = sqlite3.connect('$DB_NAME')
    cursor = conn.cursor()

    with open('hourly_weather_backup.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            cursor.execute('''
            INSERT INTO HourlyWeather (datetime, temp, feelslike, humidity, dew, precip, precipprob, snow, snowdepth, windgust, windspeed, winddir, pressure, cloudcover, visibility, solarradiation, solarenergy, uvindex, severerisk, conditions, icon, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['datetime'], row['temp'], row['feelslike'], row['humidity'], row['dew'], row['precip'],
                row['precipprob'], row['snow'], row['snowdepth'], row['windgust'], row['windspeed'], row['winddir'],
                row['pressure'], row['cloudcover'], row['visibility'], row['solarradiation'], row['solarenergy'], row['uvindex'],
                row['severerisk'], row['conditions'], row['icon'],
                datetime.strptime(row['last_updated'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
            ))

    with open('daily_weather_backup.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            cursor.execute('''
            INSERT INTO DailyWeather (sunrise_time, sunset_time, description, tempmax, tempmin, uvindex, winddir, windspeedmean, windspeed, icon, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['sunrise_time'], row['sunset_time'], row['description'], row['tempmax'], row['tempmin'],
                row['uvindex'], row['winddir'], row['windspeedmean'], row['windspeed'], row['icon'],
                datetime.strptime(row['last_updated'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
            ))

    with open('hours_backup.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            cursor.execute('''
            INSERT INTO Hours (day_id, hour, hourly_weather_id)
            VALUES (?, ?, ?)
            ''', (row['day_id'], row['hour'], row['hourly_weather_id']))

    with open('days_backup.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            cursor.execute('''
            INSERT INTO Days (date, daily_weather_id)
            VALUES (?, ?)
            ''', (row['date'], row['daily_weather_id']))

    conn.commit()
    conn.close()

import_data()
EOF

echo "Database rebuild complete."
