import os
import json

class Config:
    config = None  # Class variable to store configuration
    config_file_path = None  # Class variable to store the path to the config file

    def __init__(self, config_file='hyperparameters.json'):
        if Config.config is None:
            # Determine the absolute path to the config file
            Config.config_file_path = os.path.join(os.path.dirname(__file__), config_file)

            # Check if the config file exists
            if not os.path.exists(Config.config_file_path):
                raise FileNotFoundError(f"Configuration file not found: {Config.config_file_path}")

            # Load the configuration from the JSON file
            with open(Config.config_file_path, 'r') as f:
                Config.config = json.load(f)

    def get(self, section, key=None, default=None):
        """Retrieve a specific value from the configuration."""
        if key is None:
            # Assume 'section' is actually the top-level key
            return Config.config.get(section, default)
        else:
            return Config.config.get(section, {}).get(key, default)

    def get_section(self, section):
        """Retrieve an entire section from the configuration."""
        return Config.config.get(section, {})

    def set(self, section, option, value):
        # Existing code remains unchanged
        pass

    def save(self, path):
        """Save the current configuration to a file."""
        with open(path, 'w') as f:
            json.dump(Config.config, f, indent=4)
