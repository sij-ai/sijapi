import asyncio
import asyncpg
import yaml
from pathlib import Path
import subprocess

async def load_config():
    config_path = Path(__file__).parent.parent / 'config' / 'db.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

async def get_table_size(conn, table_name):
    return await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")

async def check_postgres_version(conn):
    return await conn.fetchval("SELECT version()")

async def replicate_table(source, target, table_name):
    print(f"Replicating {table_name} from {source['ts_id']} to {target['ts_id']}")

    source_conn = await asyncpg.connect(**{k: source[k] for k in ['db_name', 'db_user', 'db_pass', 'ts_ip', 'db_port']})
    target_conn = await asyncpg.connect(**{k: target[k] for k in ['db_name', 'db_user', 'db_pass', 'ts_ip', 'db_port']})

    try:
        source_version = await check_postgres_version(source_conn)
        target_version = await check_postgres_version(target_conn)
        print(f"Source database version: {source_version}")
        print(f"Target database version: {target_version}")

        table_size = await get_table_size(source_conn, table_name)
        print(f"Table size: {table_size} rows")

        # Dump the table
        dump_command = [
            'pg_dump',
            '-h', source['ts_ip'],
            '-p', str(source['db_port']),
            '-U', source['db_user'],
            '-d', source['db_name'],
            '-t', table_name,
            '--no-owner',
            '--no-acl'
        ]
        env = {'PGPASSWORD': source['db_pass']}
        dump_result = subprocess.run(dump_command, env=env, capture_output=True, text=True)
        
        if dump_result.returncode != 0:
            raise Exception(f"Dump failed: {dump_result.stderr}")

        print("Dump completed successfully")

        # Drop and recreate the table on the target
        await target_conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
        print(f"Dropped table {table_name} on target")

        # Restore the table
        restore_command = [
            'psql',
            '-h', target['ts_ip'],
            '-p', str(target['db_port']),
            '-U', target['db_user'],
            '-d', target['db_name'],
        ]
        env = {'PGPASSWORD': target['db_pass']}
        restore_result = subprocess.run(restore_command, input=dump_result.stdout, env=env, capture_output=True, text=True)
        
        if restore_result.returncode != 0:
            raise Exception(f"Restore failed: {restore_result.stderr}")

        print(f"Table {table_name} restored successfully")

        # Verify the number of rows in the target table
        target_size = await get_table_size(target_conn, table_name)
        if target_size == table_size:
            print(f"Replication successful. {target_size} rows copied.")
        else:
            print(f"Warning: Source had {table_size} rows, but target has {target_size} rows.")

    except Exception as e:
        print(f"An error occurred while replicating {table_name}: {str(e)}")
    finally:
        await source_conn.close()
        await target_conn.close()

async def main():
    config = await load_config()
    source_server = config['POOL'][0]  # sij-mbp16
    target_servers = config['POOL'][1:]  # sij-vm and sij-vps

    tables_to_replicate = [
        'click_logs', 'dailyweather', 'hourlyweather', 'locations', 'short_urls'
    ]

    for table_name in tables_to_replicate:
        for target_server in target_servers:
            await replicate_table(source_server, target_server, table_name)

    print("All replications completed!")

if __name__ == "__main__":
    asyncio.run(main())
