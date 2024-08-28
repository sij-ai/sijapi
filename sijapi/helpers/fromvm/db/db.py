import asyncio
import asyncpg

# Database connection information
DB_INFO = {
	'host': '100.64.64.20',
	'port': 5432,
	'database': 'sij',
	'user': 'sij',
	'password': 'Synchr0!'
}

async def update_click_logs():
	# Connect to the database
	conn = await asyncpg.connect(**DB_INFO)

	try:
		# Drop existing 'id' and 'new_id' columns if they exist
		await conn.execute("""
			ALTER TABLE click_logs
			DROP COLUMN IF EXISTS id,
			DROP COLUMN IF EXISTS new_id;
		""")
		print("Dropped existing id and new_id columns (if they existed)")

		# Add new UUID column as primary key
		await conn.execute("""
			ALTER TABLE click_logs
			ADD COLUMN id UUID PRIMARY KEY DEFAULT gen_random_uuid();
		""")
		print("Added new UUID column as primary key")

		# Get the number of rows in the table
		row_count = await conn.fetchval("SELECT COUNT(*) FROM click_logs")
		print(f"Number of rows in click_logs: {row_count}")

	except Exception as e:
		print(f"An error occurred: {str(e)}")
		import traceback
		traceback.print_exc()
	finally:
		# Close the database connection
		await conn.close()

# Run the update
asyncio.run(update_click_logs())
