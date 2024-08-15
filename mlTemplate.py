import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "mlProject"


list_of_files = [
    f"src/data/__init__.py",
    f"src/preprocess/__init__.py",
    f"src/model/__init__.py",
    f"src/utils/__init__.py",
    f"src/pipelines/__init__.py",
    f"test/unit/base.py",
    f"test/integration/base.py",
    f"test/system/base.py",
    "main.py",
    "app.py",
    "requirements.txt",
    "readme.md",
    ".gitignore",
    "experiments/base.ipynb",

]




for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")