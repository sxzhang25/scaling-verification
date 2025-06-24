import json
import os
from typing import Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy 

@dataclass
class VerifierConfig:
    name: str
    enabled: bool
    params: Dict[str, Any]

class VerifierHandler:
    """Handles loading and validation of verifier configurations."""
    
    def __init__(self, config_path: str):
        """
        Initialize configuration handler.
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            
        self._validate_config(config)
        return config
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration structure.
        
        Args:
            config: Loaded configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
            
        if 'verifiers' not in config:
            raise ValueError("Configuration must contain 'verifiers' key")
            
        if not isinstance(config['verifiers'], list):
            raise ValueError("'verifiers' must be a list")
            
        required_verifier_keys = {'name', 'enabled', 'params'}
        for verifier in config['verifiers']:
            if not isinstance(verifier, dict):
                raise ValueError("Each verifier must be a dictionary")
                
            missing_keys = required_verifier_keys - set(verifier.keys())
            if missing_keys:
                raise ValueError(f"Verifier missing required keys: {missing_keys}")
    
    def get_enabled_verifiers(self) -> List[VerifierConfig]:
        """Get list of enabled verifier configurations."""
        return [
            VerifierConfig(
                name=v['name'],
                enabled=v['enabled'],
                params=v['params']
            )
            for v in self.config['verifiers']
            if v['enabled']
        ]
    
    def get_global_params(self) -> Dict[str, Any]:
        """Get global parameters from configuration."""
        return self.config.get('global_params', {})
    
    def get_config_no_verifiers(self):
        """Get configuration with no verifiers. Used for caching"""
        config = deepcopy(self.config)
        config['verifiers'] = []
        return config