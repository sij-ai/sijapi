import psycopg2
from psycopg2 import sql

def connect_to_db():
	return psycopg2.connect(
		dbname='sij',
		user='sij',
		password='Synchr0!',
		host='localhost'  # Adjust if your database is not on localhost
	)

def get_table_info(conn):
	with conn.cursor() as cur:
		# Get all tables in the public schema
		cur.execute("""
			SELECT table_name 
			FROM information_schema.tables 
			WHERE table_schema = 'public'
		""")
		tables = cur.fetchall()

		table_info = {}
		for (table_name,) in tables:
			table_info[table_name] = {
				'primary_keys': get_primary_keys(cur, table_name),
				'foreign_keys': get_foreign_keys(cur, table_name)
			}

	return table_info

def get_primary_keys(cur, table_name):
	cur.execute("""
		SELECT a.attname
		FROM   pg_index i
		JOIN   pg_attribute a ON a.attrelid = i.indrelid
							 AND a.attnum = ANY(i.indkey)
		WHERE  i.indrelid = %s::regclass
		AND    i.indisprimary
	""", (table_name,))
	return [row[0] for row in cur.fetchall()]

def get_foreign_keys(cur, table_name):
	cur.execute("""
		SELECT
			tc.constraint_name, 
			kcu.column_name, 
			ccu.table_name AS foreign_table_name,
			ccu.column_name AS foreign_column_name 
		FROM 
			information_schema.table_constraints AS tc 
			JOIN information_schema.key_column_usage AS kcu
			  ON tc.constraint_name = kcu.constraint_name
			  AND tc.table_schema = kcu.table_schema
			JOIN information_schema.constraint_column_usage AS ccu
			  ON ccu.constraint_name = tc.constraint_name
			  AND ccu.table_schema = tc.table_schema
		WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name=%s
	""", (table_name,))
	return cur.fetchall()

def main():
	try:
		with connect_to_db() as conn:
			table_info = get_table_info(conn)

			for table_name, info in table_info.items():
				print(f"\n## Table: {table_name}")
				
				print("\nPrimary Keys:")
				if info['primary_keys']:
					for pk in info['primary_keys']:
						print(f"- {pk}")
				else:
					print("- No primary keys found")
				
				print("\nForeign Keys:")
				if info['foreign_keys']:
					for fk in info['foreign_keys']:
						print(f"- {fk[1]} -> {fk[2]}.{fk[3]} (Constraint: {fk[0]})")
				else:
					print("- No foreign keys found")

	except psycopg2.Error as e:
		print(f"Database error: {e}")
	except Exception as e:
		print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
	main()
