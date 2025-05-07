import os
import importlib

def import_all_nn_modules():
    folder = os.path.dirname(__file__)
    for filename in os.listdir(folder):
        if filename.endswith(".py") and filename != "__init__.py":
            importlib.import_module(f"{__name__}.{filename[:-3]}")

import_all_nn_modules()
