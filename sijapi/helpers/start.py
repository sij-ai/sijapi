import yaml
import requests
import paramiko
import time
from pathlib import Path
import logging
import subprocess
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    config_path = Path(__file__).parent.parent / 'config' / 'api.yaml'
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
    logging.info(f"Checking {address} for response...")
    try:
        response = requests.get(address, timeout=5)
        response_text = response.text.strip().strip('"') 
        logging.info(f"{address} responded '{response_text}'")
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

def start_local_server(server):
    try:
        if is_local_tmux_session_running('sijapi'):
            logging.info("Local sijapi tmux session is already running.")
            return

        command = f"{server['tmux']} new-session -d -s sijapi 'cd {server['path']} && {server['conda_env']}/bin/python -m sijapi'"
        logging.info(f"Executing local command: {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logging.info(f"Successfully started sijapi session on local machine")
        logging.debug(f"Command output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to start sijapi session on local machine. Error: {e}")
        logging.error(f"Error output: {e.stderr}")

def start_remote_server(server):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(
            server['ts_ip'],
            port=server['ssh_port'],
            username=server['ssh_user'],
            password=server['ssh_pass'],
            timeout=10
        )
        
        # Check if tmux session already exists
        status, output, error = execute_ssh_command(ssh, f"{server['tmux']} has-session -t sijapi 2>/dev/null && echo 'exists' || echo 'not exists'")
        if output == 'exists':
            logging.info(f"sijapi session already exists on {server['ts_id']}")
            return

        command = f"{server['tmux']} new-session -d -s sijapi 'cd {server['path']} && {server['conda_env']}/bin/python -m sijapi'"
        status, output, error = execute_ssh_command(ssh, command)
        
        if status == 0:
            logging.info(f"Successfully started sijapi session on {server['ts_id']}")
        else:
            logging.error(f"Failed to start sijapi session on {server['ts_id']}. Error: {error}")
        
    except paramiko.SSHException as e:
        logging.error(f"Failed to connect to {server['ts_id']}: {str(e)}")
    finally:
        ssh.close()



def main():
    load_env()
    config = load_config()
    pool = config['POOL']
    local_ts_id = os.environ.get('TS_ID')
    
    for server in pool:
        logging.info(f"Checking {server['ts_id']}...")
        if check_server(server['ts_ip'], server['app_port'], server['ts_id']):
            logging.info(f"{server['ts_id']} is running and responding correctly.")
        else:
            logging.info(f"{server['ts_id']} is not responding correctly. Attempting to start...")
            if server['ts_id'] == local_ts_id:
                start_local_server(server)
            else:
                start_remote_server(server)
        
        logging.info("Waiting 5 seconds before next check...")
        time.sleep(5)


if __name__ == "__main__":
    main()
