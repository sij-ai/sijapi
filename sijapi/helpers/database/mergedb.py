import sqlite3
from pathlib import Path

# Get the home directory
home_dir = Path.home()

# Define the path to the database
DB = home_dir / "sync" / "sijapi" / "data" / "weatherlocate.db"

def create_database():
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()

        # Create the Locations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                street TEXT,
                city TEXT,
                state TEXT,
                country TEXT,
                latitude REAL,
                longitude REAL,
                zip TEXT,
                elevation REAL,
                last_updated DATETIME
            );
        ''')

        # Create the Days table with a direct reference to DailyWeather
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Days (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE NOT NULL,
                daily_weather_id INTEGER,
                general_location_id INTEGER,
                FOREIGN KEY(daily_weather_id) REFERENCES DailyWeather(id),
                FOREIGN KEY(general_location_id) REFERENCES Locations(id)
            );
        ''')

        # Create the DailyWeather table with fields adjusted for direct CSV storage of preciptype
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS DailyWeather (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sunrise TEXT,
                sunriseEpoch TEXT,
                sunset TEXT,
                sunsetEpoch TEXT,
                description TEXT,
                tempmax REAL,
                tempmin REAL,
                uvindex INTEGER,
                winddir REAL,
                windspeed REAL,
                icon TEXT,
                last_updated DATETIME,
                datetime TEXT,
                datetimeEpoch INTEGER,
                temp REAL,
                feelslikemax REAL,
                feelslikemin REAL,
                feelslike REAL,
                dew REAL,
                humidity REAL,
                precip REAL,
                precipprob REAL,
                precipcover REAL,
                preciptype TEXT,
                snow REAL,
                snowdepth REAL,
                windgust REAL,
                pressure REAL,
                cloudcover REAL,
                visibility REAL,
                solarradiation REAL,
                solarenergy REAL,
                severerisk REAL,
                moonphase REAL,
                conditions TEXT,
                stations TEXT,
                source TEXT
            );
        ''')

        # Create the HourlyWeather table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS HourlyWeather (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day_id INTEGER,
                datetime TEXT,
                datetimeEpoch INTEGER,
                temp REAL,
                feelslike REAL,
                humidity REAL,
                dew REAL,
                precip REAL,
                precipprob REAL,
                snow REAL,
                snowdepth REAL,
                preciptype TEXT,
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
                stations TEXT,
                source TEXT,
                FOREIGN KEY(day_id) REFERENCES Days(id)
            );
        ''')

        conn.commit()

if __name__ == "__main__":
    create_database()
