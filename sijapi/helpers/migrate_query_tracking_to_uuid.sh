#!/bin/bash

# PostgreSQL connection details
DB_NAME="sij"
DB_USER="sij"
DB_PASSWORD="Synchr0!"
DB_HOST="localhost"
DB_PORT="5432"

# Function to execute SQL commands
execute_sql() {
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "$1"
}

# Main script
echo "Starting migration of query_tracking table..."

# Enable uuid-ossp extension if not already enabled
execute_sql "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"

# Add a new UUID column
execute_sql "ALTER TABLE query_tracking ADD COLUMN new_id UUID DEFAULT uuid_generate_v4();"

# Generate new UUIDs for all existing rows
execute_sql "UPDATE query_tracking SET new_id = uuid_generate_v4() WHERE new_id IS NULL;"

# Drop the old id column and rename the new one
execute_sql "ALTER TABLE query_tracking DROP COLUMN id;"
execute_sql "ALTER TABLE query_tracking RENAME COLUMN new_id TO id;"

# Set the new id column as primary key
execute_sql "ALTER TABLE query_tracking ADD PRIMARY KEY (id);"

echo "Migration completed successfully!"

