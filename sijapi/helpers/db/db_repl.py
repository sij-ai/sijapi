import yaml
import subprocess
import os
import sys

def load_config():
    with open('../config/sys.yaml', 'r') as file:
        return yaml.safe_load(file)

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()

def pg_dump(host, port, db_name, user, password, tables):
    dump_command = f"PGPASSWORD={password} pg_dump -h {host} -p {port} -U {user} -d {db_name} -t {' -t '.join(tables)} -c --no-owner"
    return run_command(dump_command)

def pg_restore(host, port, db_name, user, password, dump_data):
    restore_command = f"PGPASSWORD={password} psql -h {host} -p {port} -U {user} -d {db_name}"
    process = subprocess.Popen(restore_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate(input=dump_data.encode())
    return process.returncode, stdout.decode(), stderr.decode()

def check_postgres_version(host, port, user, password):
    version_command = f"PGPASSWORD={password} psql -h {host} -p {port} -U {user} -c 'SELECT version();'"
    returncode, stdout, stderr = run_command(version_command)
    if returncode == 0:
        return stdout.strip()
    else:
        return f"Error checking version: {stderr}"

def replicate_databases():
    config = load_config()
    pool = config['POOL']
    tables_to_replicate = ['click_logs', 'dailyweather', 'hourlyweather', 'locations', 'short_urls']

    source_db = pool[0]
    target_dbs = pool[1:]

    # Check source database version
    source_version = check_postgres_version(source_db['ts_ip'], source_db['db_port'], source_db['db_user'], source_db['db_pass'])
    print(f"Source database version: {source_version}")

    for target_db in target_dbs:
        print(f"\nReplicating to {target_db['ts_id']}...")
        
        # Check target database version
        target_version = check_postgres_version(target_db['ts_ip'], target_db['db_port'], target_db['db_user'], target_db['db_pass'])
        print(f"Target database version: {target_version}")

        # Perform dump
        returncode, dump_data, stderr = pg_dump(
            source_db['ts_ip'],
            source_db['db_port'],
            source_db['db_name'],
            source_db['db_user'],
            source_db['db_pass'],
            tables_to_replicate
        )

        if returncode != 0:
            print(f"Error during dump: {stderr}")
            continue

        # Perform restore
        returncode, stdout, stderr = pg_restore(
            target_db['ts_ip'],
            target_db['db_port'],
            target_db['db_name'],
            target_db['db_user'],
            target_db['db_pass'],
            dump_data
        )

        if returncode != 0:
            print(f"Error during restore: {stderr}")
        else:
            print(f"Replication to {target_db['ts_id']} completed successfully.")

if __name__ == "__main__":
    replicate_databases()

