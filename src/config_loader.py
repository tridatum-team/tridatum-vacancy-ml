import os
import socket
import yaml
from copy import deepcopy

def get_current_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsocketname()[0]
        s.close()
        return ip
    except:
        return None

def detect_env():
    ip = get_current_ip()
    if ip:
        if '208' in ip:
            return 'prod'
        elif '197' in ip or '199' in ip:
            return 'dev'

def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result

def load_config(config_path='configs/config.yaml', env=None):
    if env is None or env == 'None':
        env = os.environ.get('PIPELINE_ENV') or detect_env()

    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)

    common = raw_config.get('common', {})
    env_config = raw_config.get(env, {})

    config = deep_merge(common, env_config)
    config['environment'] = env

    print(f"[CONFIG] Loaded: {env} (IP: {get_current_ip()})")
    return config

    
