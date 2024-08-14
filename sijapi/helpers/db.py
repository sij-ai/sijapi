import asyncio
import asyncpg
import yaml
from pathlib import Path

async def load_config():
    config_path = Path(__file__).parent.parent / 'config' / 'sys.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

async def add_foreign_key_constraint(conn):
    # Ensure short_code is not null in both tables
    await conn.execute("""
    ALTER TABLE short_urls
    ALTER COLUMN short_code SET NOT NULL;
    """)

    await conn.execute("""
    ALTER TABLE click_logs
    ALTER COLUMN short_code SET NOT NULL;
    """)

    # Add unique constraint to short_urls.short_code if it doesn't exist
    await conn.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conname = 'short_urls_short_code_key'
        ) THEN
            ALTER TABLE short_urls
            ADD CONSTRAINT short_urls_short_code_key UNIQUE (short_code);
        END IF;
    END $$;
    """)

    # Add foreign key constraint
    await conn.execute("""
    ALTER TABLE click_logs
    ADD CONSTRAINT fk_click_logs_short_urls
    FOREIGN KEY (short_code)
    REFERENCES short_urls(short_code)
    ON DELETE CASCADE;
    """)

    print("Foreign key constraint added successfully.")

async def main():
    config = await load_config()
    source_server = config['POOL'][0]  # sij-mbp16

    conn_params = {
        'database': source_server['db_name'],
        'user': source_server['db_user'],
        'password': source_server['db_pass'],
        'host': source_server['ts_ip'],
        'port': source_server['db_port']
    }

    conn = await asyncpg.connect(**conn_params)

    try:
        await add_foreign_key_constraint(conn)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
