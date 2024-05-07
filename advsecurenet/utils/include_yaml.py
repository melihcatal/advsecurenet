import os

from ruamel.yaml import YAML


class IncludeYAML:
    def __init__(self):
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.indent(mapping=2, sequence=4, offset=2)
        self.yaml.Constructor.add_constructor(
            "!include", self.include_constructor)
        self.current_path = None  # Tracks the path of the current file being processed
