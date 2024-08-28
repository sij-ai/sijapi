#!/usr/bin/env python3

import os
import yaml
import subprocess
import time
from tqdm import tqdm

def load_config():
	script_dir = os.path.dirname(os.path.abspath(__file__))
	sys_config_path = os.path.join(script_dir, '..', 'config', 'sys.yaml')
	gis_config_path = os.path.join(script_dir, '..', 'config', 'gis.yaml')

	with open(sys_config_path, 'r') as f:
		sys_config = yaml.safe_load(f)

	with open(gis_config_path, 'r') as f:
		gis_config = yaml.safe_load(f)

	return sys_config, gis_config

def get_table_size(server, table_name):
	env = os.environ.copy()
	env['PGPASSWORD'] = server['db_pass']
	
	command = [
		'psql',
		'-h', server['ts_ip'],
		'-p', str(server['db_port']),
		'-U', server['db_user'],
		'-d', server['db_name'],
		'-t',
		'-c', f"SELECT COUNT(*) FROM {table_name}"
	]
	
	result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
	return int(result.stdout.strip())

def replicate_table(source, targets, table_name):
	print(f"Replicating {table_name}")

	# Get table size for progress bar
	table_size = get_table_size(source, table_name)
	print(f"Table size: {table_size} rows")

	# Dump the table from the source
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
	
	env = os.environ.copy()
	env['PGPASSWORD'] = source['db_pass']

	print("Dumping table...")
	with open(f"{table_name}.sql", 'w') as f:
		subprocess.run(dump_command, env=env, stdout=f, check=True)
	print("Dump complete")

	# Restore the table to each target
	for target in targets:
		print(f"Replicating to {target['ts_id']}")
		
		# Drop table and its sequence
		drop_commands = [
			f"DROP TABLE IF EXISTS {table_name} CASCADE;",
			f"DROP SEQUENCE IF EXISTS {table_name}_id_seq CASCADE;"
		]
		
		restore_command = [
			'psql',
			'-h', target['ts_ip'],
			'-p', str(target['db_port']),
			'-U', target['db_user'],
			'-d', target['db_name'],
		]
		
		env = os.environ.copy()
		env['PGPASSWORD'] = target['db_pass']

		# Execute drop commands
		for cmd in drop_commands:
			print(f"Executing: {cmd}")
			subprocess.run(restore_command + ['-c', cmd], env=env, check=True)

		# Restore the table
		print("Restoring table...")
		process = subprocess.Popen(restore_command + ['-f', f"{table_name}.sql"], env=env, 
								   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

		pbar = tqdm(total=table_size, desc="Copying rows")
		copied_rows = 0
		for line in process.stderr:
			if line.startswith("COPY"):
				copied_rows = int(line.split()[1])
				pbar.update(copied_rows - pbar.n)
			print(line, end='')  # Print all output for visibility

		pbar.close()
		process.wait()

		if process.returncode != 0:
			print(f"Error occurred during restoration to {target['ts_id']}")
			print(process.stderr.read())
		else:
			print(f"Restoration to {target['ts_id']} completed successfully")

	# Clean up the dump file
	os.remove(f"{table_name}.sql")
	print(f"Replication of {table_name} completed")

def main():
	sys_config, gis_config = load_config()

	source_server = sys_config['POOL'][0]
	target_servers = sys_config['POOL'][1:]

	tables = [layer['table_name'] for layer in gis_config['layers']]

	for table in tables:
		replicate_table(source_server, target_servers, table)

	print("All replications completed!")

if __name__ == "__main__":
	main()
