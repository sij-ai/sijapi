#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Set variables
DB_NAME="sij"
DB_USER="sij"
OSM_FILE="north-america-latest.osm.pbf"
FLAT_NODES="/Users/sij/workshop/sijapi/sijapi/data/db/flat-nodes.bin"

# Ensure the directory for flat-nodes exists
mkdir -p "$(dirname "$FLAT_NODES")"

# Determine total system memory in MB
TOTAL_MEM=$(sysctl hw.memsize | awk '{print $2 / 1024 / 1024}')

# Calculate cache size (50% of total memory, max 32GB)
CACHE_SIZE=$(echo "scale=0; $TOTAL_MEM * 0.5 / 1" | bc)
CACHE_SIZE=$(( CACHE_SIZE > 32768 ? 32768 : CACHE_SIZE ))

# Calculate number of processes (number of CPU cores minus 1, min 1)
NUM_PROCESSES=$(sysctl -n hw.ncpu)
NUM_PROCESSES=$(( NUM_PROCESSES > 1 ? NUM_PROCESSES - 1 : 1 ))

echo "Starting OSM data import..."

# Run osm2pgsql
osm2pgsql -d $DB_NAME \
          --create \
          --slim \
          -G \
          --hstore \
          --tag-transform-script /opt/homebrew/Cellar/osm2pgsql/1.11.0_1/share/osm2pgsql/openstreetmap-carto.lua \
          -C $CACHE_SIZE \
          --number-processes $NUM_PROCESSES \
          -S /opt/homebrew/Cellar/osm2pgsql/1.11.0_1/share/osm2pgsql/default.style \
          --prefix osm \
          -H localhost \
          -P 5432 \
          -U $DB_USER \
          --flat-nodes $FLAT_NODES \
          $OSM_FILE

echo "OSM data import completed. Creating indexes..."

# Create indexes (adjust table names if necessary)
psql -d $DB_NAME -U $DB_USER -c "CREATE INDEX IF NOT EXISTS idx_osm_point_way ON osm_point USING GIST (way);"
psql -d $DB_NAME -U $DB_USER -c "CREATE INDEX IF NOT EXISTS idx_osm_line_way ON osm_line USING GIST (way);"
psql -d $DB_NAME -U $DB_USER -c "CREATE INDEX IF NOT EXISTS idx_osm_polygon_way ON osm_polygon USING GIST (way);"

echo "Import completed and indexes created."
