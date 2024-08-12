#!/bin/bash

# Configuration
SOURCE_HOST="100.64.64.20"
SOURCE_PORT="5432"
SOURCE_DB="sij"
SOURCE_USER="sij"
SOURCE_PASS="Synchr0!"

# Target servers
declare -a TARGETS=(
  "sij-vm:100.64.64.11:5432:sij:sij:Synchr0!"
  "sij-vps:100.64.64.15:5432:sij:sij:Synchr0!"
)

# Tables to replicate
TABLES=("dailyweather" "hourlyweather" "short_urls" "click_logs" "locations")

# PostgreSQL binaries
PSQL="/Applications/Postgres.app/Contents/Versions/latest/bin/psql"
PG_DUMP="/Applications/Postgres.app/Contents/Versions/latest/bin/pg_dump"

# Function to run SQL and display results
run_sql() {
    local host=$1
    local port=$2
    local db=$3
    local user=$4
    local pass=$5
    local sql=$6

    PGPASSWORD=$pass $PSQL -h $host -p $port -U $user -d $db -c "$sql"
}

# Replicate to a target
replicate_to_target() {
    local target_info=$1
    IFS=':' read -r target_name target_host target_port target_db target_user target_pass <<< "$target_info"

    echo "Replicating to $target_name ($target_host)"

    # Check source tables
    echo "Checking source tables:"
    for table in "${TABLES[@]}"; do
        run_sql $SOURCE_HOST $SOURCE_PORT $SOURCE_DB $SOURCE_USER $SOURCE_PASS "SELECT COUNT(*) FROM $table;"
    done

    # Dump and restore each table
    for table in "${TABLES[@]}"; do
        echo "Replicating $table"
        
        # Dump table
        PGPASSWORD=$SOURCE_PASS $PG_DUMP -h $SOURCE_HOST -p $SOURCE_PORT -U $SOURCE_USER -d $SOURCE_DB -t $table --no-owner --no-acl > ${table}_dump.sql
        
        if [ $? -ne 0 ]; then
            echo "Error dumping $table"
            continue
        fi

        # Drop and recreate table on target
        run_sql $target_host $target_port $target_db $target_user $target_pass "DROP TABLE IF EXISTS $table CASCADE; "
        
        # Restore table
        PGPASSWORD=$target_pass $PSQL -h $target_host -p $target_port -U $target_user -d $target_db -f ${table}_dump.sql
        
        if [ $? -ne 0 ]; then
            echo "Error restoring $table"
        else
            echo "$table replicated successfully"
        fi

        # Clean up dump file
        rm ${table}_dump.sql
    done

    # Verify replication
    echo "Verifying replication:"
    for table in "${TABLES[@]}"; do
        echo "Checking $table on target:"
        run_sql $target_host $target_port $target_db $target_user $target_pass "SELECT COUNT(*) FROM $table;"
    done
}

# Main replication process
for target in "${TARGETS[@]}"; do
    replicate_to_target "$target"
done

echo "Replication completed"

