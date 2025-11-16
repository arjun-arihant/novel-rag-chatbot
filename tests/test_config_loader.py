"""
Tests for configuration loading
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config_loader import load_config, get_config_value, update_config


class TestConfigLoader(unittest.TestCase):
    """Test configuration loading."""

    def test_load_config(self):
        """Test loading config file."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

        if os.path.exists(config_path):
            config = load_config(config_path)
            self.assertIsInstance(config, dict)
            self.assertIn('models', config)

    def test_get_config_value(self):
        """Test getting nested config value."""
        config = {
            'models': {
                'llm': 'mistral:7b'
            }
        }

        value = get_config_value(config, 'models', 'llm')
        self.assertEqual(value, 'mistral:7b')

        # Test default value
        value = get_config_value(config, 'nonexistent', default='default')
        self.assertEqual(value, 'default')

    def test_update_config(self):
        """Test updating config."""
        config = {'a': {'b': 1}}
        updates = {'a': {'c': 2}}

        updated = update_config(config, updates)
        self.assertEqual(updated['a']['b'], 1)
        self.assertEqual(updated['a']['c'], 2)


if __name__ == '__main__':
    unittest.main()
