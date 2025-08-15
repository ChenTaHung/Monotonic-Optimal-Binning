"""Configuration settings for MOBPY.

This module provides global configuration options that affect the behavior
of the entire package. Users can modify these settings to customize MOBPY
for their specific use cases.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json
import os
from pathlib import Path


@dataclass
class MOBPYConfig:
    """Global configuration for MOBPY package.
    
    Controls package-wide behavior including numerical tolerances,
    display options, and performance settings.
    
    Attributes:
        epsilon: Numerical tolerance for floating point comparisons.
        max_iterations: Maximum iterations for iterative algorithms.
        enable_progress_bar: Whether to show progress bars in long operations.
        plot_style: Default matplotlib style for plots.
        n_jobs: Number of parallel jobs (-1 for all CPUs).
        random_state: Random seed for reproducibility.
        display_precision: Number of decimal places in displays.
        warn_on_small_bins: Whether to warn when bins have few samples.
        small_bin_threshold: Threshold for small bin warnings.
    """
    
    # Numerical settings
    epsilon: float = 1e-12
    max_iterations: int = 1000
    
    # Display settings  
    enable_progress_bar: bool = False
    plot_style: str = "seaborn-v0_8-darkgrid"
    display_precision: int = 4
    
    # Performance settings
    n_jobs: int = 1
    random_state: Optional[int] = None
    
    # Warning settings
    warn_on_small_bins: bool = True
    small_bin_threshold: int = 30
    
    # Advanced settings
    cache_intermediate_results: bool = False
    validate_inputs: bool = True
    
    # Private fields
    _custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def set(self, **kwargs) -> None:
        """Update configuration values.
        
        Args:
            **kwargs: Configuration parameters to update.
            
        Raises:
            AttributeError: If trying to set non-existent parameter.
            
        Examples:
            >>> config = MOBPYConfig()
            >>> config.set(epsilon=1e-10, n_jobs=-1)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Configuration has no parameter '{key}'")
    
    def reset(self) -> None:
        """Reset all configuration to default values."""
        self.__init__()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary.
        
        Returns:
            Dict containing all configuration parameters.
        """
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
    
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary.
        
        Args:
            config_dict: Dictionary of configuration parameters.
        """
        self.set(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration file.
            
        Examples:
            >>> config.save("my_config.json")
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file.
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            
        Examples:
            >>> config.load("my_config.json")
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        self.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, filepath: str) -> "MOBPYConfig":
        """Create configuration instance from file.
        
        Args:
            filepath: Path to configuration file.
            
        Returns:
            MOBPYConfig: New configuration instance.
        """
        config = cls()
        config.load(filepath)
        return config
    
    @classmethod
    def from_env(cls) -> "MOBPYConfig":
        """Create configuration from environment variables.
        
        Looks for environment variables prefixed with MOBPY_.
        
        Returns:
            MOBPYConfig: Configuration with values from environment.
            
        Examples:
            >>> # Set environment variable
            >>> os.environ["MOBPY_EPSILON"] = "1e-10"
            >>> config = MOBPYConfig.from_env()
        """
        config = cls()
        
        for key in dir(config):
            if key.startswith('_'):
                continue
            
            env_key = f"MOBPY_{key.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                
                # Type conversion based on current type
                current_value = getattr(config, key)
                if isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes')
                elif isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                
                setattr(config, key, value)
        
        return config


# Global configuration instance
_global_config = MOBPYConfig()


def get_config() -> MOBPYConfig:
    """Get the global MOBPY configuration.
    
    Returns:
        MOBPYConfig: Global configuration instance.
        
    Examples:
        >>> from MOBPY.config import get_config
        >>> config = get_config()
        >>> config.set(epsilon=1e-8)
    """
    return _global_config


def set_config(**kwargs) -> None:
    """Update global configuration.
    
    Args:
        **kwargs: Configuration parameters to update.
        
    Examples:
        >>> from MOBPY.config import set_config
        >>> set_config(epsilon=1e-10, n_jobs=4)
    """
    _global_config.set(**kwargs)


def reset_config() -> None:
    """Reset global configuration to defaults.
    
    Examples:
        >>> from MOBPY.config import reset_config
        >>> reset_config()
    """
    _global_config.reset()