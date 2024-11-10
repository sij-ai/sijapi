import yaml
import requests
import paramiko
import time
from pathlib import Path
import logging
import subprocess
import os
import argparse
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(cfg: str):
    config_path = Path(__file__).parent.parent / 'config' / f'{cfg}.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_env():
    env_path = Path(__file__).parent.parent / 'config' / '.env'
    if env_path.exists():
        with open(env_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
                    except ValueError:
                        logging.warning(f"Skipping invalid line in .env file: {line}")

def check_server(ip, port, ts_id):
    address = f"http://{ip}:{port}/id"
    try:
        response = requests.get(address, timeout=5)
        response_text = response.text.strip().strip('"')
        return response.status_code == 200 and response_text == ts_id
    except requests.RequestException as e:
        logging.error(f"Error checking server {ts_id}: {str(e)}")
        return False

def execute_ssh_command(ssh, command):
    stdin, stdout, stderr = ssh.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    output = stdout.read().decode().strip()
    error = stderr.read().decode().strip()
    return exit_status, output, error

def is_local_tmux_session_running(session_name):
    try:
        result = subprocess.run(['tmux', 'has-session', '-t', session_name], capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False

def start_local_server(server, pull=False, push=False):
    try:
        if is_local_tmux_session_running('sijapi'):
            logging.info("Local sijapi tmux session is already running.")
            return

        git_command = ""
        if pull:
            git_command = "git pull &&"
        elif push:
            git_command = "git add -A . && git commit -m \"auto-update\" && git push origin --force &&"

        command = f"{server['tmux']} new-session -d -s sijapi 'cd {server['path']} && {git_command} {server['conda_env']}/bin/python -m sijapi'"
        logging.info(f"Executing local command: {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logging.info(f"Successfully started sijapi session on local machine")
        logging.debug(f"Command output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to start sijapi session on local machine. Error: {e}")
        logging.error(f"Error output: {e.stderr}")

def kill_local_server():
    try:
        if is_local_tmux_session_running('sijapi'):
            subprocess.run(['tmux', 'kill-session', '-t', 'sijapi'], check=True)
            logging.info("Killed local sijapi tmux session.")
        else:
            logging.info("No local sijapi tmux session to kill.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to kill local sijapi tmux session. Error: {e}")

def start_remote_server(server, pull=False, push=False):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Determine authentication method based on config
        if 'ssh_key' in server:
            # Use SSH key authentication
            private_key = paramiko.RSAKey.from_private_key_file(server['ssh_key'])
            ssh.connect(
                server['ts_ip'],
                port=server['ssh_port'],
                username=server['ssh_user'],
                pkey=private_key,
                timeout=10
            )
        elif 'ssh_pass' in server:
            # Use password authentication
            ssh.connect(
                server['ts_ip'],
                port=server['ssh_port'],
                username=server['ssh_user'],
                password=server['ssh_pass'],
                timeout=10
            )
        else:
            logging.error(f"No authentication method specified for {server['ts_id']}")
            return

        status, output, error = execute_ssh_command(ssh, f"{server['tmux']} has-session -t sijapi 2>/dev/null && echo 'exists' || echo 'not exists'")
        if output == 'exists':
            logging.info(f"sijapi session already exists on {server['ts_id']}")
            return

        git_command = ""
        if pull:
            git_command = "git pull &&"
        elif push:
            git_command = "git add -A . && git commit -m \"auto-update\" && git push origin --force &&"

        command = f"{server['tmux']} new-session -d -s sijapi 'cd {server['path']} && {git_command} {server['conda_env']}/bin/python -m sijapi'"
        status, output, error = execute_ssh_command(ssh, command)

        if status == 0:
            logging.info(f"Successfully started sijapi session on {server['ts_id']}")
        else:
            logging.error(f"Failed to start sijapi session on {server['ts_id']}. Error: {error}")

    except paramiko.SSHException as e:
        logging.error(f"Failed to connect to {server['ts_id']}: {str(e)}")
    except Exception as e:
        logging.error(f"Error connecting to {server['ts_id']}: {str(e)}")
    finally:
        ssh.close()

def kill_remote_server(server):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Determine authentication method based on config
        if 'ssh_key' in server:
            # Use SSH key authentication
            private_key = paramiko.RSAKey.from_private_key_file(server['ssh_key'])
            ssh.connect(
                server['ts_ip'],
                port=server['ssh_port'],
                username=server['ssh_user'],
                pkey=private_key,
                timeout=10
            )
        elif 'ssh_pass' in server:
            # Use password authentication
            ssh.connect(
                server['ts_ip'],
                port=server['ssh_port'],
                username=server['ssh_user'],
                password=server['ssh_pass'],
                timeout=10
            )
        else:
            logging.error(f"No authentication method specified for {server['ts_id']}")
            return

        command = f"{server['tmux']} kill-session -t sijapi"
        status, output, error = execute_ssh_command(ssh, command)

        if status == 0:
            logging.info(f"Successfully killed sijapi session on {server['ts_id']}")
        else:
            logging.error(f"Failed to kill sijapi session on {server['ts_id']}. Error: {error}")

    except paramiko.SSHException as e:
        logging.error(f"Failed to connect to {server['ts_id']}: {str(e)}")
    except Exception as e:
        logging.error(f"Error connecting to {server['ts_id']}: {str(e)}")
    finally:
        ssh.close()


def main():
    load_env()
    db_config = load_config('sys')
    pool = db_config['POOL']
    local_ts_id = os.environ.get('TS_ID')

    parser = argparse.ArgumentParser(description='Manage sijapi servers')
    parser.add_argument('--kill', action='store_true', help='Kill the local sijapi tmux session')
    parser.add_argument('--restart', action='store_true', help='Restart the local sijapi tmux session')
    parser.add_argument('--all', action='store_true', help='Apply the action to all servers')
    parser.add_argument('--pull', action='store_true', help='Pull latest changes before starting the server')
    parser.add_argument('--push', action='store_true', help='Push changes before starting the server')

    args = parser.parse_args()

    if args.kill:
        if args.all:
            for server in pool:
                if server['ts_id'] == local_ts_id:
                    kill_local_server()
                else:
                    kill_remote_server(server)
        else:
            kill_local_server()
        sys.exit(0)

    if args.restart or args.pull or args.push:
        if args.all:
            for server in pool:
                if server['ts_id'] == local_ts_id:
                    kill_local_server()
                    start_local_server(server, pull=args.pull, push=args.push)
                else:
                    kill_remote_server(server)
                    start_remote_server(server, pull=args.pull, push=args.push)
        else:
            kill_local_server()
            local_server = next(server for server in pool if server['ts_id'] == local_ts_id)
            start_local_server(local_server, pull=args.pull, push=args.push)
        sys.exit(0)

    # If no specific arguments, run the default behavior
    local_server = next(server for server in pool if server['ts_id'] == local_ts_id)
    if not check_server(local_server['ts_ip'], local_server['app_port'], local_server['ts_id']):
        logging.info(f"Local server {local_server['ts_id']} is not responding correctly. Attempting to start...")
        kill_local_server()
        start_local_server(local_server, push=True)

    for server in pool:
        if server['ts_id'] != local_ts_id:
            if not check_server(server['ts_ip'], server['app_port'], server['ts_id']):
                logging.info(f"{server['ts_id']} is not responding correctly. Attempting to start...")
                kill_remote_server(server)
                start_remote_server(server, pull=True)
            else:
                logging.info(f"{server['ts_id']} is running and responding correctly.")

        time.sleep(1)

if __name__ == "__main__":
    main()