# modules/config_loader.py
import os
import yaml

def load_config(config_path="cfgs/gui/paths/default.yaml"):
    """
    Loads the configuration from a YAML file and returns it as a dictionary.
    """
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    else:
        raise FileNotFoundError(f"Config file {config_path} not found.")
