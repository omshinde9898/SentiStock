import yaml

def read_yaml_config(filepath:str) -> dict:
    with open(filepath,'r+') as file:
        content = yaml.safe_load(file)
    return content