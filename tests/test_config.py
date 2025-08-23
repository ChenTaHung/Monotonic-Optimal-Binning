"""Unit tests for configuration module.

This module tests the global configuration system including
settings management, persistence, and environment variable loading.
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pickle

from MOBPY.config import (
    MOBPYConfig, get_config, set_config, reset_config
)


class TestMOBPYConfig:
    """Test suite for MOBPYConfig class.
    
    Tests configuration initialization, modification, and persistence.
    """
    
    def test_default_initialization(self):
        """Test default configuration values."""
        config = MOBPYConfig()
        
        # Numerical settings
        assert config.epsilon == 1e-12
        assert config.max_iterations == 1000
        
        # Display settings
        assert config.enable_progress_bar is False
        assert config.plot_style == "seaborn-v0_8-darkgrid"
        assert config.display_precision == 4
        
        # Performance settings
        assert config.n_jobs == 1
        assert config.random_state is None
        
        # Warning settings
        assert config.warn_on_small_bins is True
        assert config.small_bin_threshold == 30
        
        # Advanced settings
        assert config.cache_intermediate_results is False
        assert config.validate_inputs is True
    
    def test_set_method(self):
        """Test setting configuration values."""
        config = MOBPYConfig()
        
        # Set single value
        config.set(epsilon=1e-10)
        assert config.epsilon == 1e-10
        
        # Set multiple values
        config.set(
            max_iterations=500,
            n_jobs=4,
            enable_progress_bar=True
        )
        assert config.max_iterations == 500
        assert config.n_jobs == 4
        assert config.enable_progress_bar is True
    
    def test_set_invalid_parameter(self):
        """Test setting non-existent parameter raises error."""
        config = MOBPYConfig()
        
        with pytest.raises(AttributeError, match="no parameter"):
            config.set(invalid_param=123)
    
    def test_reset_method(self):
        """Test resetting configuration to defaults."""
        config = MOBPYConfig()
        
        # Modify settings
        config.set(
            epsilon=1e-8,
            max_iterations=100,
            n_jobs=-1
        )
        
        # Reset
        config.reset()
        
        # Should be back to defaults
        assert config.epsilon == 1e-12
        assert config.max_iterations == 1000
        assert config.n_jobs == 1
    
    def test_to_dict_method(self):
        """Test exporting configuration as dictionary."""
        config = MOBPYConfig()
        config.set(epsilon=1e-10, n_jobs=2)
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['epsilon'] == 1e-10
        assert config_dict['n_jobs'] == 2
        
        # Should not include private attributes
        assert '_custom_settings' not in config_dict
    
    def test_from_dict_method(self):
        """Test loading configuration from dictionary."""
        config = MOBPYConfig()
        
        config_dict = {
            'epsilon': 1e-8,
            'max_iterations': 200,
            'n_jobs': 4,
            'enable_progress_bar': True
        }
        
        config.from_dict(config_dict)
        
        assert config.epsilon == 1e-8
        assert config.max_iterations == 200
        assert config.n_jobs == 4
        assert config.enable_progress_bar is True
    
    def test_save_and_load_json(self):
        """Test saving and loading configuration to/from JSON file."""
        config = MOBPYConfig()
        config.set(
            epsilon=1e-9,
            max_iterations=250,
            plot_style="ggplot"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "config.json")
            
            # Save
            config.save(filepath)
            assert os.path.exists(filepath)
            
            # Load into new config
            new_config = MOBPYConfig()
            new_config.load(filepath)
            
            assert new_config.epsilon == 1e-9
            assert new_config.max_iterations == 250
            assert new_config.plot_style == "ggplot"
    
    def test_save_creates_directories(self):
        """Test that save creates parent directories if needed."""
        config = MOBPYConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Nested path that doesn't exist
            filepath = os.path.join(tmpdir, "subdir", "nested", "config.json")
            
            config.save(filepath)
            
            assert os.path.exists(filepath)
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file raises error."""
        config = MOBPYConfig()
        
        with pytest.raises(FileNotFoundError):
            config.load("nonexistent_config.json")
    
    def test_from_file_class_method(self):
        """Test creating config instance from file."""
        original = MOBPYConfig()
        original.set(epsilon=1e-11, n_jobs=3)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "config.json")
            original.save(filepath)
            
            # Create new instance from file
            loaded = MOBPYConfig.from_file(filepath)
            
            assert loaded.epsilon == 1e-11
            assert loaded.n_jobs == 3
    
    def test_from_env_class_method(self):
        """Test creating config from environment variables."""
        # Set environment variables
        env_vars = {
            'MOBPY_EPSILON': '1e-10',
            'MOBPY_MAX_ITERATIONS': '500',
            'MOBPY_N_JOBS': '4',
            'MOBPY_ENABLE_PROGRESS_BAR': 'true',
            'MOBPY_WARN_ON_SMALL_BINS': 'false',
            'MOBPY_DISPLAY_PRECISION': '6'
        }
        
        with patch.dict(os.environ, env_vars):
            config = MOBPYConfig.from_env()
            
            assert config.epsilon == 1e-10
            assert config.max_iterations == 500
            assert config.n_jobs == 4
            assert config.enable_progress_bar is True
            assert config.warn_on_small_bins is False
            assert config.display_precision == 6
    
    def test_from_env_type_conversion(self):
        """Test type conversion when loading from environment."""
        env_vars = {
            'MOBPY_EPSILON': '0.001',  # Float
            'MOBPY_MAX_ITERATIONS': '1500',  # Int
            'MOBPY_ENABLE_PROGRESS_BAR': '1',  # Bool (using 1)
            'MOBPY_WARN_ON_SMALL_BINS': 'yes',  # Bool (using yes)
            'MOBPY_CACHE_INTERMEDIATE_RESULTS': 'FALSE',  # Bool (uppercase)
        }
        
        with patch.dict(os.environ, env_vars):
            config = MOBPYConfig.from_env()
            
            assert config.epsilon == 0.001
            assert isinstance(config.epsilon, float)
            
            assert config.max_iterations == 1500
            assert isinstance(config.max_iterations, int)
            
            assert config.enable_progress_bar is True
            assert config.warn_on_small_bins is True
            assert config.cache_intermediate_results is False
    
    def test_from_env_ignores_non_mobpy_vars(self):
        """Test that from_env ignores non-MOBPY environment variables."""
        env_vars = {
            'MOBPY_EPSILON': '1e-9',
            'OTHER_VAR': 'value',
            'PATH': '/usr/bin'
        }
        
        with patch.dict(os.environ, env_vars):
            config = MOBPYConfig.from_env()
            
            assert config.epsilon == 1e-9
            # Other vars should not affect config
            assert not hasattr(config, 'OTHER_VAR')
            assert not hasattr(config, 'PATH')


class TestGlobalConfig:
    """Test suite for global configuration functions."""
    
    def test_get_config_returns_singleton(self):
        """Test that get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_set_config_updates_global(self):
        """Test set_config updates global configuration."""
        # Store original values
        original_epsilon = get_config().epsilon
        
        try:
            # Update global config
            set_config(epsilon=1e-7, n_jobs=2)
            
            config = get_config()
            assert config.epsilon == 1e-7
            assert config.n_jobs == 2
        finally:
            # Restore original
            set_config(epsilon=original_epsilon, n_jobs=1)
    
    def test_reset_config_resets_global(self):
        """Test reset_config resets global configuration."""
        # Store original values
        original_config = get_config().to_dict()
        
        try:
            # Modify global config
            set_config(
                epsilon=1e-5,
                max_iterations=50,
                n_jobs=-1
            )
            
            # Reset
            reset_config()
            
            config = get_config()
            assert config.epsilon == 1e-12  # Default
            assert config.max_iterations == 1000  # Default
            assert config.n_jobs == 1  # Default
        finally:
            # Restore original
            get_config().from_dict(original_config)


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_config_affects_mobpy_behavior(self):
        """Test that configuration actually affects MOBPY behavior."""
        from MOBPY.core.pava import PAVA
        import pandas as pd
        import numpy as np
        
        # Create test data
        df = pd.DataFrame({
            'x': range(100),
            'y': np.random.random(100)
        })
        
        original_config = get_config().to_dict()
        
        try:
            # Set very low iteration limit
            set_config(max_iterations=2)
            
            # This should still work but may not fully converge
            # Use keyword arguments for PAVA initialization
            pava = PAVA(df=df, x='x', y='y')
            pava.fit()  # Should complete even with low iterations
            
        finally:
            get_config().from_dict(original_config)
    
    def test_config_serialization_roundtrip(self):
        """Test configuration can be serialized and restored."""
        config = MOBPYConfig()
        config.set(
            epsilon=1e-11,
            max_iterations=333,
            n_jobs=5,
            plot_style="custom_style"
        )
        
        # Test JSON roundtrip
        config_dict = config.to_dict()
        new_config = MOBPYConfig()
        new_config.from_dict(config_dict)
        
        assert new_config.epsilon == config.epsilon
        assert new_config.max_iterations == config.max_iterations
        assert new_config.n_jobs == config.n_jobs
        assert new_config.plot_style == config.plot_style
    
    def test_config_pickle_serialization(self):
        """Test configuration can be pickled."""
        config = MOBPYConfig()
        config.set(epsilon=1e-13, n_jobs=7)
        
        # Pickle and unpickle
        pickled = pickle.dumps(config)
        restored = pickle.loads(pickled)
        
        assert restored.epsilon == 1e-13
        assert restored.n_jobs == 7
    
    def test_config_validation_ranges(self):
        """Test configuration value validation."""
        config = MOBPYConfig()
        
        # Epsilon should be positive
        config.set(epsilon=1e-15)  # Very small but positive - OK
        assert config.epsilon == 1e-15
        
        # Max iterations should be positive
        config.set(max_iterations=1)  # Minimum valid value
        assert config.max_iterations == 1
        
        # n_jobs can be -1 (all CPUs)
        config.set(n_jobs=-1)
        assert config.n_jobs == -1
        
        # Display precision should be non-negative
        config.set(display_precision=0)
        assert config.display_precision == 0
    
    def test_config_context_manager_pattern(self):
        """Test using configuration in context manager pattern."""
        original_config = get_config().to_dict()
        
        try:
            original_epsilon = get_config().epsilon
            
            # Temporarily change config
            set_config(epsilon=1e-6)
            assert get_config().epsilon == 1e-6
            
            # Would be nice to have context manager for temporary changes
            # This is a suggestion for future enhancement
            
            # Restore
            set_config(epsilon=original_epsilon)
            assert get_config().epsilon == original_epsilon
            
        finally:
            get_config().from_dict(original_config)
    
    def test_config_thread_safety_considerations(self):
        """Test configuration behavior with threading considerations."""
        # Note: Current implementation uses global singleton
        # This test documents current behavior
        
        config = get_config()
        original_epsilon = config.epsilon
        
        try:
            # Changes affect global state
            set_config(epsilon=1e-8)
            
            # All references see the change
            assert get_config().epsilon == 1e-8
            
            # This is important to document for users running parallel code
            
        finally:
            set_config(epsilon=original_epsilon)