import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging
from pathlib import Path

@dataclass
class AIServiceConfig:
    """Configuration for individual AI services"""
    url: str
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 2
    health_check_interval: int = 300  # 5 minutes
    api_key: Optional[str] = None

@dataclass
class TradingParameters:
    """Trading-specific parameters"""
    max_position_size: float = 0.1  # 10% of portfolio
    risk_per_trade: float = 0.02    # 2% risk per trade
    default_leverage: int = 10
    min_confidence_threshold: float = 0.6
    high_risk_confidence_threshold: float = 0.8
    max_daily_trades: int = 50
    max_concurrent_positions: int = 10
    
    # Risk management
    max_portfolio_risk: float = 0.2  # 20% total portfolio risk
    correlation_limit: float = 0.8   # Max correlation between positions
    volatility_limit: float = 100    # Max volatility threshold
    
    # Position management
    trailing_stop_activation: float = 0.02  # 2% profit before trailing
    default_take_profit_ratio: float = 2.0  # 2:1 reward to risk
    max_position_duration_hours: int = 168  # 1 week

@dataclass 
class DatabaseConfig:
    """Database configuration"""
    path: str = "conversations.db"
    backup_interval_hours: int = 24
    cleanup_age_days: int = 7
    max_conversation_records: int = 10000

@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    enable_prometheus: bool = True
    prometheus_port: int = 8000
    health_check_port: int = 8080
    log_level: str = "INFO"
    log_file: str = "trading_system.log"
    alert_webhook_url: Optional[str] = None
    
    # Alert thresholds
    max_consecutive_failures: int = 5
    high_risk_position_threshold: float = 0.1  # 10% of portfolio
    low_confidence_alert_threshold: float = 0.4

@dataclass
class SecurityConfig:
    """Security configuration"""
    api_key: str = field(default_factory=lambda: os.getenv('API_KEY', ''))
    api_secret: str = field(default_factory=lambda: os.getenv('API_SECRET', ''))
    webhook_secret: str = field(default_factory=lambda: os.getenv('WEBHOOK_SECRET', ''))
    enable_request_signing: bool = True
    rate_limit_calls: int = 100
    rate_limit_period: int = 3600  # 1 hour

class EnhancedConfig:
    """Enhanced configuration management system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or os.getenv('CONFIG_FILE', 'config.yaml')
        self.load_config()
    
    def load_config(self):
        """Load configuration from environment and files"""
        
        # AI Services Configuration
        self.ai_services = {
            'main_strategy': AIServiceConfig(
                url=os.getenv('IA_STRATEGY_URL', 'http://localhost:8001/api/analyze'),
                timeout=int(os.getenv('IA_STRATEGY_TIMEOUT', '30')),
                max_retries=int(os.getenv('IA_STRATEGY_RETRIES', '3')),
                api_key=os.getenv('IA_STRATEGY_API_KEY')
            ),
            'specialized': AIServiceConfig(
                url=os.getenv('IA_SPECIALIZED_URL', 'http://localhost:8002/api/consult'),
                timeout=int(os.getenv('IA_SPECIALIZED_TIMEOUT', '45')),
                max_retries=int(os.getenv('IA_SPECIALIZED_RETRIES', '3')),
                api_key=os.getenv('IA_SPECIALIZED_API_KEY')
            ),
            'bingx': AIServiceConfig(
                url=os.getenv('BINGX_IA_URL', 'http://localhost:8003/api/position'),
                timeout=int(os.getenv('BINGX_IA_TIMEOUT', '30')),
                max_retries=int(os.getenv('BINGX_IA_RETRIES', '3')),
                api_key=os.getenv('BINGX_API_KEY')
            )
        }
        
        # Trading Parameters
        self.trading = TradingParameters(
            max_position_size=float(os.getenv('MAX_POSITION_SIZE', '0.1')),
            risk_per_trade=float(os.getenv('RISK_PER_TRADE', '0.02')),
            default_leverage=int(os.getenv('DEFAULT_LEVERAGE', '10')),
            min_confidence_threshold=float(os.getenv('MIN_CONFIDENCE', '0.6')),
            max_daily_trades=int(os.getenv('MAX_DAILY_TRADES', '50')),
            max_concurrent_positions=int(os.getenv('MAX_CONCURRENT_POSITIONS', '10'))
        )
        
        # Database Configuration
        self.database = DatabaseConfig(
            path=os.getenv('DATABASE_PATH', 'conversations.db'),
            backup_interval_hours=int(os.getenv('DB_BACKUP_INTERVAL', '24')),
            cleanup_age_days=int(os.getenv('DB_CLEANUP_DAYS', '7'))
        )
        
        # Monitoring Configuration
        self.monitoring = MonitoringConfig(
            enable_prometheus=os.getenv('ENABLE_PROMETHEUS', 'true').lower() == 'true',
            prometheus_port=int(os.getenv('PROMETHEUS_PORT', '8000')),
            health_check_port=int(os.getenv('HEALTH_PORT', '8080')),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_file=os.getenv('LOG_FILE', 'trading_system.log'),
            alert_webhook_url=os.getenv('ALERT_WEBHOOK_URL')
        )
        
        # Security Configuration
        self.security = SecurityConfig(
            api_key=os.getenv('API_KEY', ''),
            api_secret=os.getenv('API_SECRET', ''),
            webhook_secret=os.getenv('WEBHOOK_SECRET', ''),
            enable_request_signing=os.getenv('ENABLE_SIGNING', 'true').lower() == 'true',
            rate_limit_calls=int(os.getenv('RATE_LIMIT_CALLS', '100')),
            rate_limit_period=int(os.getenv('RATE_LIMIT_PERIOD', '3600'))
        )
        
        # Application Configuration
        self.app = {
            'name': os.getenv('APP_NAME', 'Advanced Trading Orchestrator'),
            'version': os.getenv('APP_VERSION', '2.0.0'),
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'debug': os.getenv('DEBUG', 'false').lower() == 'true',
            'cycle_interval_seconds': int(os.getenv('CYCLE_INTERVAL', '300')),  # 5 minutes
            'startup_delay_seconds': int(os.getenv('STARTUP_DELAY', '10'))
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate required API keys
        if not self.security.api_key:
            errors.append("API_KEY is required")
        
        if not self.security.api_secret:
            errors.append("API_SECRET is required")
        
        # Validate AI service URLs
        for service_name, config in self.ai_services.items():
            if not config.url or not config.url.startswith(('http://', 'https://')):
                errors.append(f"Invalid URL for {service_name}: {config.url}")
        
        # Validate trading parameters
        if not 0 < self.trading.max_position_size <= 1:
            errors.append(f"Invalid max_position_size: {self.trading.max_position_size}")
        
        if not 0 < self.trading.risk_per_trade <= 0.1:
            errors.append(f"Invalid risk_per_trade: {self.trading.risk_per_trade}")
        
        # Validate file paths
        db_dir = Path(self.database.path).parent
        if not db_dir.exists():
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create database directory: {e}")
        
        return errors
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.monitoring.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.monitoring.log_file),
                logging.StreamHandler()
            ]
        )
        
        # Reduce noise from external libraries
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    def get_ai_service_config(self, service_name: str) -> AIServiceConfig:
        """Get configuration for specific AI service"""
        return self.ai_services.get(service_name)
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.app['environment'] == 'production'
    
    def get_cache_settings(self) -> Dict:
        """Get cache configuration settings"""
        return {
            'ttl': int(os.getenv('CACHE_TTL', '300')),  # 5 minutes
            'maxsize': int(os.getenv('CACHE_MAXSIZE', '100')),
            'enabled': os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        }
    
    def get_circuit_breaker_settings(self) -> Dict:
        """Get circuit breaker configuration"""
        return {
            'failure_threshold': int(os.getenv('CB_FAILURE_THRESHOLD', '5')),
            'recovery_timeout': int(os.getenv('CB_RECOVERY_TIMEOUT', '60')),
            'expected_exception': Exception
        }
    
    def export_env_template(self, filepath: str = '.env.template'):
        """Export environment variables template"""
        template_content = '''# Advanced Trading Orchestrator Configuration

# AI Service URLs
IA_STRATEGY_URL=http://localhost:8001/api/analyze
IA_SPECIALIZED_URL=http://localhost:8002/api/consult
BINGX_IA_URL=http://localhost:8003/api/position

# API Keys and Secrets
API_KEY=your_api_key_here
API_SECRET=your_api_secret_here
WEBHOOK_SECRET=your_webhook_secret_here

# Trading Parameters
MAX_POSITION_SIZE=0.1
RISK_PER_TRADE=0.02
DEFAULT_LEVERAGE=10
MIN_CONFIDENCE=0.6
MAX_DAILY_TRADES=50
MAX_CONCURRENT_POSITIONS=10

# Database Configuration
DATABASE_PATH=conversations.db
DB_BACKUP_INTERVAL=24
DB_CLEANUP_DAYS=7

# Monitoring Configuration
ENABLE_PROMETHEUS=true
PROMETHEUS_PORT=8000
HEALTH_PORT=8080
LOG_LEVEL=INFO
LOG_FILE=trading_system.log

# Application Configuration
APP_NAME=Advanced Trading Orchestrator
APP_VERSION=2.0.0
ENVIRONMENT=development
DEBUG=false
CYCLE_INTERVAL=300

# Cache Configuration
CACHE_TTL=300
CACHE_MAXSIZE=100
CACHE_ENABLED=true

# Circuit Breaker Configuration
CB_FAILURE_THRESHOLD=5
CB_RECOVERY_TIMEOUT=60

# Rate Limiting
RATE_LIMIT_CALLS=100
RATE_LIMIT_PERIOD=3600

# Startup Configuration
STARTUP_DELAY=10
'''
        
        with open(filepath, 'w') as f:
            f.write(template_content)

# Global configuration instance
config = EnhancedConfig()