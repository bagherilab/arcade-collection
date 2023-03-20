import importlib
import os
import sys

from prefect import task

for module_file in os.listdir(os.path.dirname(__file__)):
    if "__" in module_file or not module_file.endswith(".py"):
        continue

    module_name = module_file.replace(".py", "")

    module = importlib.import_module(f".{module_name}", package=__name__)
    setattr(sys.modules[__name__], module_name, task(getattr(module, module_name)))
