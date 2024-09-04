import yaml
import os

class Settings:
    _instance = None

    def __new__(cls, env='base'):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._initialize(env)
        return cls._instance

    def _initialize(self, env):
        self.config = self.load_config(env)
        self._populate_attributes(self.config)

    def load_config(self, env):
        base_config_path = os.path.join(os.path.dirname(__file__), '../..', 'conf', 'base.yaml')
        env_config_path = os.path.join(os.path.dirname(__file__), '../..', 'conf', f'{env}.yaml')

        base_config = self.load_yaml(base_config_path)
        if os.path.exists(env_config_path):
            env_config = self.load_yaml(env_config_path)
            config = self.merge_dicts(base_config, env_config)
        else:
            config = base_config

        return config

    def load_yaml(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def merge_dicts(self, base_dict, override_dict):
        """Recursively merge two dictionaries."""
        for key, value in override_dict.items():
            if isinstance(value, dict) and key in base_dict:
                base_dict[key] = self.merge_dicts(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict

    def _populate_attributes(self, config):
        """Populate object attributes from the configuration dictionary."""
        for key, value in config.items():
            if isinstance(value, dict):
                # Create a nested attribute for dict values
                sub_attr = type('ConfigSection', (), {})()
                setattr(self, key, sub_attr)
                self._populate_attributes_recursive(value, sub_attr)
            else:
                setattr(self, key, value)

    def _populate_attributes_recursive(self, config, parent_obj):
        """Recursive helper to populate nested attributes."""
        for key, value in config.items():
            if isinstance(value, dict):
                # Create a nested attribute for dict values
                sub_attr = type('ConfigSection', (), {})()
                setattr(parent_obj, key, sub_attr)
                self._populate_attributes_recursive(value, sub_attr)
            else:
                setattr(parent_obj, key, value)

# Initialize the settings object with the default environment
settings = Settings(env='base')
