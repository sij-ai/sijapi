import psycopg2
from psycopg2 import sql
import sys

def connect_to_db():
	return psycopg2.connect(
		dbname='sij',
		user='sij',
		password='Synchr0!',
		host='localhost'
	)

def get_tables(cur):
	cur.execute("""
		SELECT table_name 
		FROM information_schema.tables 
		WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
		AND table_name NOT LIKE '%_uuid' AND table_name NOT LIKE '%_orig'
		AND table_name != 'spatial_ref_sys'
	""")
	return [row[0] for row in cur.fetchall()]

def get_columns(cur, table_name):
	cur.execute("""
		SELECT column_name, udt_name, 
			   is_nullable, column_default,
			   character_maximum_length, numeric_precision, numeric_scale
		FROM information_schema.columns 
		WHERE table_name = %s
		ORDER BY ordinal_position
	""", (table_name,))
	return cur.fetchall()

def get_constraints(cur, table_name):
	cur.execute("""
		SELECT conname, contype, pg_get_constraintdef(c.oid)
		FROM pg_constraint c
		JOIN pg_namespace n ON n.oid = c.connamespace
		WHERE conrelid = %s::regclass
		AND n.nspname = 'public'
	""", (table_name,))
	return cur.fetchall()

def drop_table_if_exists(cur, table_name):
	cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table_name)))

def create_uuid_table(cur, old_table, new_table):
	drop_table_if_exists(cur, new_table)
	columns = get_columns(cur, old_table)
	constraints = get_constraints(cur, old_table)
	
	column_defs = []
	has_id_column = any(col[0] == 'id' for col in columns)
	
	for col in columns:
		col_name, udt_name, is_nullable, default, max_length, precision, scale = col
		if col_name == 'id' and has_id_column:
			column_defs.append(sql.SQL("{} UUID PRIMARY KEY DEFAULT gen_random_uuid()").format(sql.Identifier(col_name)))
		else:
			type_sql = sql.SQL("{}").format(sql.Identifier(udt_name))
			if max_length:
				type_sql = sql.SQL("{}({})").format(type_sql, sql.Literal(max_length))
			elif precision and scale:
				type_sql = sql.SQL("{}({},{})").format(type_sql, sql.Literal(precision), sql.Literal(scale))
			
			column_def = sql.SQL("{} {}").format(sql.Identifier(col_name), type_sql)
			if is_nullable == 'NO':
				column_def = sql.SQL("{} NOT NULL").format(column_def)
			if default and 'nextval' not in default:  # Skip auto-increment defaults
				column_def = sql.SQL("{} DEFAULT {}").format(column_def, sql.SQL(default))
			column_defs.append(column_def)

	constraint_defs = []
	for constraint in constraints:
		conname, contype, condef = constraint
		if contype != 'p' or not has_id_column:  # Keep primary key if there's no id column
			constraint_defs.append(sql.SQL(condef))

	if not has_id_column:
		column_defs.append(sql.SQL("uuid UUID DEFAULT gen_random_uuid()"))

	query = sql.SQL("CREATE TABLE {} ({})").format(
		sql.Identifier(new_table),
		sql.SQL(", ").join(column_defs + constraint_defs)
	)
	cur.execute(query)

def migrate_data(cur, old_table, new_table):
	columns = get_columns(cur, old_table)
	column_names = [col[0] for col in columns]
	has_id_column = 'id' in column_names
	
	if has_id_column:
		column_names.remove('id')
		old_cols = sql.SQL(", ").join(map(sql.Identifier, column_names))
		new_cols = sql.SQL(", ").join(map(sql.Identifier, ['id'] + column_names))
		query = sql.SQL("INSERT INTO {} ({}) SELECT gen_random_uuid(), {} FROM {}").format(
			sql.Identifier(new_table),
			new_cols,
			old_cols,
			sql.Identifier(old_table)
		)
	else:
		old_cols = sql.SQL(", ").join(map(sql.Identifier, column_names))
		new_cols = sql.SQL(", ").join(map(sql.Identifier, column_names + ['uuid']))
		query = sql.SQL("INSERT INTO {} ({}) SELECT {}, gen_random_uuid() FROM {}").format(
			sql.Identifier(new_table),
			new_cols,
			old_cols,
			sql.Identifier(old_table)
		)
	cur.execute(query)

def update_foreign_keys(cur, tables):
	for table in tables:
		constraints = get_constraints(cur, table)
		for constraint in constraints:
			conname, contype, condef = constraint
			if contype == 'f':  # Foreign key constraint
				referenced_table = condef.split('REFERENCES ')[1].split('(')[0].strip()
				referenced_column = condef.split('(')[2].split(')')[0].strip()
				local_column = condef.split('(')[1].split(')')[0].strip()
				
				cur.execute(sql.SQL("""
					UPDATE {table_uuid}
					SET {local_column} = subquery.new_id::text::{local_column_type}
					FROM (
						SELECT old.{ref_column} AS old_id, new_table.id AS new_id
						FROM {ref_table} old
						JOIN public.{ref_table_uuid} new_table ON new_table.{ref_column}::text = old.{ref_column}::text
					) AS subquery
					WHERE {local_column}::text = subquery.old_id::text
				""").format(
					table_uuid=sql.Identifier(f"{table}_uuid"),
					local_column=sql.Identifier(local_column),
					local_column_type=sql.SQL(get_column_type(cur, f"{table}_uuid", local_column)),
					ref_column=sql.Identifier(referenced_column),
					ref_table=sql.Identifier(referenced_table),
					ref_table_uuid=sql.Identifier(f"{referenced_table}_uuid")
				))

def get_column_type(cur, table_name, column_name):
	cur.execute("""
		SELECT data_type 
		FROM information_schema.columns 
		WHERE table_name = %s AND column_name = %s
	""", (table_name, column_name))
	return cur.fetchone()[0]

def rename_tables(cur, tables):
	for table in tables:
		drop_table_if_exists(cur, f"{table}_orig")
		cur.execute(sql.SQL("ALTER TABLE IF EXISTS {} RENAME TO {}").format(
			sql.Identifier(table), sql.Identifier(f"{table}_orig")
		))
		cur.execute(sql.SQL("ALTER TABLE IF EXISTS {} RENAME TO {}").format(
			sql.Identifier(f"{table}_uuid"), sql.Identifier(table)
		))

def main():
	try:
		with connect_to_db() as conn:
			with conn.cursor() as cur:
				tables = get_tables(cur)
				
				# Create new UUID tables
				for table in tables:
					print(f"Creating UUID table for {table}...")
					create_uuid_table(cur, table, f"{table}_uuid")
				
				# Migrate data
				for table in tables:
					print(f"Migrating data for {table}...")
					migrate_data(cur, table, f"{table}_uuid")
				
				# Update foreign keys
				print("Updating foreign key references...")
				update_foreign_keys(cur, tables)
				
				# Rename tables
				print("Renaming tables...")
				rename_tables(cur, tables)
				
			conn.commit()
		print("Migration completed successfully.")
	except Exception as e:
		print(f"An error occurred: {e}")
		conn.rollback()

if __name__ == "__main__":
	main()
