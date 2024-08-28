# helpers/replicator.py

import asyncio
import asyncpg
import yaml
from pathlib import Path
import subprocess
import sys
import os

async def load_config():
    config_path = Path(__file__).parent.parent / 'config' / 'db.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

async def check_table_existence(conn, tables):
    for table in tables:
        exists = await conn.fetchval(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = $1
            )
        """, table)
        print(f"Table {table} {'exists' if exists else 'does not exist'} in the database.")

async def check_user_permissions(conn, tables):
    for table in tables:
        has_permission = await conn.fetchval(f"""
            SELECT has_table_privilege(current_user, $1, 'SELECT')
        """, table)
        print(f"User {'has' if has_permission else 'does not have'} SELECT permission on table {table}.")

async def replicate_tables(source, target, tables):
    print(f"Replicating tables from {source['ts_id']} to {target['ts_id']}")

    conn_params = {
        'database': 'db_name',
        'user': 'db_user',
        'password': 'db_pass',
        'host': 'ts_ip',
        'port': 'db_port'
    }

    source_conn = await asyncpg.connect(**{k: source[v] for k, v in conn_params.items()})
    target_conn = await asyncpg.connect(**{k: target[v] for k, v in conn_params.items()})

    try:
        source_version = await source_conn.fetchval("SELECT version()")
        target_version = await target_conn.fetchval("SELECT version()")
        print(f"Source database version: {source_version}")
        print(f"Target database version: {target_version}")

        print("Checking table existence in source database:")
        await check_table_existence(source_conn, tables)

        print("\nChecking user permissions in source database:")
        await check_user_permissions(source_conn, tables)

        # Dump all tables to a file
        dump_file = 'dump.sql'
        dump_command = [
            '/Applications/Postgres.app/Contents/Versions/latest/bin/pg_dump',
            '-h', source['ts_ip'],
            '-p', str(source['db_port']),
            '-U', source['db_user'],
            '-d', source['db_name'],
            '-t', ' -t '.join(tables),
            '--no-owner',
            '--no-acl',
            '-f', dump_file
        ]
        env = {'PGPASSWORD': source['db_pass']}
        print(f"\nExecuting dump command: {' '.join(dump_command)}")
        dump_result = subprocess.run(dump_command, env=env, capture_output=True, text=True)
        
        if dump_result.returncode != 0:
            print(f"Dump stderr: {dump_result.stderr}")
            raise Exception(f"Dump failed: {dump_result.stderr}")

        print("Dump completed successfully.")

        # Restore from the dump file
        restore_command = [
            '/Applications/Postgres.app/Contents/Versions/latest/bin/psql',
            '-h', target['ts_ip'],
            '-p', str(target['db_port']),
            '-U', target['db_user'],
            '-d', target['db_name'],
            '-f', dump_file
        ]
        env = {'PGPASSWORD': target['db_pass']}
        print(f"\nExecuting restore command: {' '.join(restore_command)}")
        restore_result = subprocess.run(restore_command, env=env, capture_output=True, text=True)

        if restore_result.returncode != 0:
            print(f"Restore stderr: {restore_result.stderr}")
            raise Exception(f"Restore failed: {restore_result.stderr}")

        print("Restore completed successfully.")

        # Clean up the dump file
        os.remove(dump_file)

    except Exception as e:
        print(f"An error occurred during replication: {str(e)}")
        print("Exception details:", sys.exc_info())
    finally:
        await source_conn.close()
        await target_conn.close()

async def main():
    config = await load_config()
    source_server = config['POOL'][0]  # sij-mbp16
    target_servers = config['POOL'][1:]  # sij-vm and sij-vps

    tables_to_replicate = [
        'dailyweather', 'hourlyweather', 'short_urls', 'click_logs', 'locations'
    ]

    for target_server in target_servers:
        await replicate_tables(source_server, target_server, tables_to_replicate)

    print("All replications completed!")

if __name__ == "__main__":
    asyncio.run(main())
