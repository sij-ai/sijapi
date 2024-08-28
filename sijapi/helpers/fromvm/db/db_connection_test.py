import asyncio
import asyncpg
import psycopg2
import sys

async def try_async_connect(host, port, user, password, database):
    try:
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        version = await conn.fetchval('SELECT version()')
        print(f"Async connection successful to {host}:{port}")
        print(f"PostgreSQL version: {version}")
        await conn.close()
        return True
    except Exception as e:
        print(f"Async connection failed to {host}:{port}")
        print(f"Error: {str(e)}")
        return False

def try_sync_connect(host, port, user, password, database):
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        cur = conn.cursor()
        cur.execute('SELECT version()')
        version = cur.fetchone()[0]
        print(f"Sync connection successful to {host}:{port}")
        print(f"PostgreSQL version: {version}")
        conn.close()
        return True
    except Exception as e:
        print(f"Sync connection failed to {host}:{port}")
        print(f"Error: {str(e)}")
        return False

async def main():
    # Database connection parameters
    port = 5432
    user = 'sij'
    password = 'Synchr0!'
    database = 'sij'

    hosts = ['100.64.64.20', '127.0.0.1', 'localhost']

    print("Attempting asynchronous connections:")
    for host in hosts:
        await try_async_connect(host, port, user, password, database)
        print()

    print("Attempting synchronous connections:")
    for host in hosts:
        try_sync_connect(host, port, user, password, database)
        print()

if __name__ == "__main__":
    asyncio.run(main())
