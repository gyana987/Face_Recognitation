"""Standalone stand-in for AI_VMS's real services/config_reader.py, same
cfg.get_str/get_int/get_float(key, fallback) interface and env-var-overrides-
config.ini precedence. DO NOT copy this file into AI_VMS -- it already has its
own services/config_reader.py; just point inference.py's import at that
instead (see README.md)."""

import configparser
import os


class ConfigManager:
    def __init__(self, config_path=None):
        config_path = config_path or os.path.join(os.path.dirname(__file__), "..", "config", "config.ini")
        self.parser = configparser.ConfigParser()
        self.parser.read(config_path)

    def get_str(self, key, fallback=None):
        return os.environ.get(key, self.parser.get("DEFAULT", key, fallback=fallback))

    def get_int(self, key, fallback=None):
        return int(self.get_str(key, fallback))

    def get_float(self, key, fallback=None):
        return float(self.get_str(key, fallback))


cfg = ConfigManager()
