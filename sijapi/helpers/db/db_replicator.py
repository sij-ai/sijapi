#!/usr/bin/env python3

import os
import yaml
import subprocess

def load_config():
	script_dir = os.path.dirname(os.path.abspath(__file__))
	sys_config_path = os.path.join(script_dir, '..', 'config', 'sys.yaml')
	gis_config_path = os.path.join(script_dir, '..', 'config', 'gis.yaml')

	with open(sys_config_path, 'r') as f:
		sys_config = yaml.safe_load(f)

	with open(gis_config_path, 'r') as f:
		gis_config = yaml.safe_load(f)

	return sys_config, gis_config

def replicate_table(source, targets, table_name):
	print(f"Replicating {table_name}")

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

	with open(f"{table_name}.sql", 'w') as f:
		subprocess.run(dump_command, env=env, stdout=f, check=True)

	# Restore the table to each target
	for target in targets:
		print(f"Replicating to {target['ts_id']}")
		restore_command = [
			'psql',
			'-h', target['ts_ip'],
			'-p', str(target['db_port']),
			'-U', target['db_user'],
			'-d', target['db_name'],
			'-c', f"DROP TABLE IF EXISTS {table_name} CASCADE;",
			'-f', f"{table_name}.sql"
		]
		
		env = os.environ.copy()
		env['PGPASSWORD'] = target['db_pass']

		subprocess.run(restore_command, env=env, check=True)

	# Clean up the dump file
	os.remove(f"{table_name}.sql")

def main():
	sys_config, gis_config = load_config()

	source_server = sys_config['POOL'][0]
	target_servers = sys_config['POOL'][1:]

	tables = [layer['table_name'] for layer in gis_config['layers']]

	for table in tables:
		replicate_table(source_server, target_servers, table)

	print("Replication complete!")

if __name__ == "__main__":
	main()
