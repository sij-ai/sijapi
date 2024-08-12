#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS postgis;
    CREATE EXTENSION IF NOT EXISTS postgis_topology;
EOSQL

# Modify pg_hba.conf to allow connections from Tailscale network
echo "host all all 100.64.64.0/24 trust" >> /var/lib/postgresql/data/pg_hba.conf

