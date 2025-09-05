# dependency_manager.py
"""
Centralized dependency management to handle optional imports gracefully
"""

import logging
import sys
from typing import Dict, Any, Optional, Callable

# Track what's available
AVAILABLE_FEATURES = {
    'tenacity': False,
    'cachetools': False, 
    'prometheus_client': False,
    'flask': False,
    'requests': False
}

# Import with fallbacks
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    AVAILABLE_FEATURES['tenacity'] = True
except ImportError:
    # Create basic retry fallback
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    stop_after_attempt = lambda x: None
    wait_exponential = lambda **kwargs: None
    retry_if_exception_type = lambda x: None

try:
    from cachetools import TTLCache
    AVAILABLE_FEATURES['cachetools'] = True
except ImportError:
    # Simple dict-based cache fallback
    class TTLCache(dict):
        def __init__(self, maxsize=100, ttl=300):
            super().__init__()
            self.maxsize = maxsize
            self.ttl = ttl
        
        def __setitem__(self, key, value):
            if len(self) >= self.maxsize:
                # Simple eviction - remove first item
                first_key = next(iter(self))
                del self[first_key]
            super().__setitem__(key, value)

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
    AVAILABLE_FEATURES['prometheus_client'] = True
except ImportError:
    # Mock prometheus classes
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def inc(self, amount=1):
            pass
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def observe(self, amount):
            pass
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def set(self, value):
            pass
    
    class CollectorRegistry:
        def __init__(self):
            pass
    
    def start_http_server(port, registry=None):
        logging.warning(f"Prometheus server not available - install prometheus_client")

try:
    from flask import Flask, jsonify, request
    AVAILABLE_FEATURES['flask'] = True
except ImportError:
    # Mock Flask for basic functionality
    class Flask:
        def __init__(self, name):
            self.name = name
            self.routes = {}
        
        def route(self, path, **kwargs):
            def decorator(func):
                self.routes[path] = func
                return func
            return decorator
        
        def run(self, **kwargs):
            logging.warning("Flask not available - health endpoints disabled")
    
    def jsonify(data):
        import json
        return json.dumps(data)
    
    class MockRequest:
        @property
        def json(self):
            return {}
    
    request = MockRequest()

try:
    import requests
    AVAILABLE_FEATURES['requests'] = True
except ImportError:
    # This is critical - we need requests
    logging.error("requests library is required but not installed")
    sys.exit(1)

# Configuration compatibility layer
def get_config_with_fallbacks():
    """Get configuration with fallbacks for missing modules"""
    try:
        from enhanced_config import config
        return config
    except ImportError:
        try:
            import config
            return config
        except ImportError:
            # Create minimal config
            return create_minimal_config()

def create_minimal_config():
    """Create minimal configuration for basic operation"""
    from dataclasses import dataclass
    import os
    
    @dataclass
    class MinimalConfig:
        # AI URLs
        IA_STRATEGY_URL = os.getenv('IA_STRATEGY_URL', 'http://localhost:8001/api/analyze')
        IA_SPECIALIZED_URL = os.getenv('IA_SPECIALIZED_URL', 'http://localhost:8002/api/consult')
        BINGX_IA_URL = os.getenv('BINGX_IA_URL', 'http://localhost:8003/api/position')
        
        # Basic parameters
        MAX_RETRY_ATTEMPTS = int(os.getenv('MAX_RETRY_ATTEMPTS', '3'))
        REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
        MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.1'))
        RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))
        
        # App config
        class app:
            cycle_interval_seconds = int(os.getenv('CYCLE_INTERVAL', '300'))
            environment = os.getenv('ENVIRONMENT', 'development')
        
        # Security config  
        class security:
            rate_limit_calls = int(os.getenv('RATE_LIMIT_CALLS', '100'))
            rate_limit_period = int(os.getenv('RATE_LIMIT_PERIOD', '3600'))
        
        # Monitoring config
        class monitoring:
            enable_prometheus = os.getenv('ENABLE_PROMETHEUS', 'false').lower() == 'true'
            health_check_port = int(os.getenv('HEALTH_PORT', '8080'))
        
        def setup_logging(self):
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        def validate_config(self):
            return []  # No validation errors for minimal config
    
    return MinimalConfig()

# Security utilities compatibility
def get_security_utils():
    """Get security utilities with fallbacks"""
    try:
        from enhanced_security_utils import execute_with_retry, call_ia_endpoint, validate_ia_response
        return execute_with_retry, call_ia_endpoint, validate_ia_response
    except ImportError:
        return create_fallback_security_utils()

def create_fallback_security_utils():
    """Create fallback security utilities"""
    import time
    
    def execute_with_retry(func: Callable, max_retries: int = 3, delay: int = 2, **kwargs) -> Any:
        """Simple retry implementation"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func(**kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))
                else:
                    logging.error(f"All {max_retries} attempts failed")
        
        raise last_exception
    
    def call_ia_endpoint(url: str, payload: Dict, api_key: str = None, timeout: int = 30):
        """Simple endpoint call"""
        import requests
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        return requests.post(url, json=payload, headers=headers, timeout=timeout)
    
    def validate_ia_response(response: Dict, expected_fields: list) -> bool:
        """Basic response validation"""
        if not response or not isinstance(response, dict):
            return False
        
        for field in expected_fields:
            if field not in response:
                logging.error(f"Missing field: {field}")
                return False
        
        return True
    
    return execute_with_retry, call_ia_endpoint, validate_ia_response

# Feature detection utilities
def log_available_features():
    """Log which features are available"""
    logger = logging.getLogger(__name__)
    
    logger.info("Feature availability:")
    for feature, available in AVAILABLE_FEATURES.items():
        status = "✓" if available else "✗"
        logger.info(f"  {status} {feature}")
    
    if not AVAILABLE_FEATURES['prometheus_client']:
        logger.warning("Prometheus metrics disabled - install prometheus_client for monitoring")
    
    if not AVAILABLE_FEATURES['flask']:
        logger.warning("Health endpoints disabled - install flask for health checks")
    
    if not AVAILABLE_FEATURES['tenacity']:
        logger.warning("Advanced retry logic disabled - install tenacity for better resilience")

def ensure_minimal_dependencies():
    """Ensure we have the absolute minimum to run"""
    critical_missing = []
    
    if not AVAILABLE_FEATURES['requests']:
        critical_missing.append('requests')
    
    if critical_missing:
        logging.error(f"Critical dependencies missing: {', '.join(critical_missing)}")
        logging.error("Install with: pip install " + ' '.join(critical_missing))
        sys.exit(1)

# Graceful degradation helpers
class FeatureFlag:
    """Simple feature flag system based on available dependencies"""
    
    @staticmethod
    def prometheus_enabled():
        return AVAILABLE_FEATURES['prometheus_client']
    
    @staticmethod
    def advanced_caching_enabled():
        return AVAILABLE_FEATURES['cachetools']
    
    @staticmethod
    def health_endpoints_enabled():
        return AVAILABLE_FEATURES['flask']
    
    @staticmethod
    def advanced_retry_enabled():
        return AVAILABLE_FEATURES['tenacity']
