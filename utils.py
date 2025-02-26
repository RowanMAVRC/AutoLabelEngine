import yaml

def load_yaml(file_path):
    """Load a YAML file from the specified path."""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
