import os
import sys
import yaml

def create_svhn_yaml(data_path):
    # Define the content of the YAML file
    yaml_content = {
        "path": data_path,
        "train": "train/images",
        "val": "test/images",
        "names": {
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9'
        }
    }

    # Write the YAML content to svhn.yaml file
    with open("svhn.yaml", "w") as yaml_file:
        yaml.dump(yaml_content, yaml_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]

    create_svhn_yaml(data_path)

