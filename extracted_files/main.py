# main_fixed.py
"""
Fixed main entry point with proper dependency management and error handling
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add current directory to path for local imports
sys.path.append(os.path.dirname(__file__))

# Import dependency manager first
from dependency_manager import (
    get_config_with_fallbacks, 
    get_security_utils,
    log_available_features,
    ensure_minimal_dependencies,
    FeatureFlag
)

# Import enhanced mocks
from enhanced_mocks import create_test_environment

class TradingSystem:
    def __init__(self, use_mocks: bool = True):
        # Ensure we have critical dependencies
        ensure_minimal_dependencies()
        
        # Log feature availability
        log_available_features()
        
        # Setup configuration with fallbacks
        self.config = get_config_with_fallbacks()
        self.config.setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Advanced Trading System...")
        
        # Validate configuration
        if hasattr(self.config, 'validate_config'):
            errors = self.config.validate_config()
            if errors:
                self.logger.error("Configuration errors found:")
                for error in errors:
                    self.logger.error(f"  - {error}")
                if self.config.app.environment == 'production':
                    sys.exit(1)
                else:
                    self.logger.warning("Continuing in development mode despite config errors")
        
        # Initialize components
        if use_mocks:
            test_env = create_test_environment()
            self.scouting_bot = test_env["scouting_bot"]
            self.bingx_api = test_env["bingx_api"]
        else:
            # In production, these would be real implementations
            raise NotImplementedError("Production implementations not provided")
        
        # Initialize security components with fallbacks
        self.execute_with_retry, self.call_ia_endpoint, self.validate_ia_response = get_security_utils()
        
        # Initialize monitoring if available
        self.monitor = None
        if FeatureFlag.health_endpoints_enabled():
            try:
                from monitoring_system import SystemMonitor
                self.monitor = SystemMonitor(self.config)
                if hasattr(self.config, 'monitoring') and self.config.monitoring.enable_prometheus:
                    self.monitor.start_prometheus_server(self.config.monitoring.prometheus_port)
                health_port = getattr(self.config.monitoring, 'health_check_port', 8080)
                self.monitor.start_health_server(health_port)
            except Exception as e:
                self.logger.warning(f"Monitoring initialization failed: {e}")
        
        # Initialize orchestrator with fallback import handling
        try:
            from enhanced_trading_orchestrator import AdvancedTradingOrchestrator, TradingConfig
            
            trading_config = TradingConfig(
                ia_strategy_url=getattr(self.config, 'IA_STRATEGY_URL', 'http://localhost:8001/api/analyze'),
                ia_specialized_url=getattr(self.config, 'IA_SPECIALIZED_URL', 'http://localhost:8002/api/consult'),
                bingx_ia_url=getattr(self.config, 'BINGX_IA_URL', 'http://localhost:8003/api/position'),
                max_retry_attempts=getattr(self.config, 'MAX_RETRY_ATTEMPTS', 3),
                request_timeout=getattr(self.config, 'REQUEST_TIMEOUT', 30),
                max_position_size=getattr(self.config, 'MAX_POSITION_SIZE', 0.1),
                risk_per_trade=getattr(self.config, 'RISK_PER_TRADE', 0.02)
            )
            
            self.orchestrator = AdvancedTradingOrchestrator(
                self.scouting_bot, 
                self.bingx_api,
                trading_config
            )
        except ImportError as e:
            self.logger.error(f"Failed to import orchestrator: {e}")
            # Create a minimal orchestrator fallback
            self.orchestrator = self._create_minimal_orchestrator()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)
        
        self.is_running = False
        self.logger.info("Trading system initialization complete")
    
    def _create_minimal_orchestrator(self):
        """Create a minimal orchestrator if the full version fails to load"""
        class MinimalOrchestrator:
            def __init__(self, scouting_bot, bingx_api):
                self.scouting_bot = scouting_bot
                self.bingx_api = bingx_api
                self.logger = logging.getLogger(self.__class__.__name__)
            
            def run_complete_trading_cycle(self):
                self.logger.info("Running minimal trading cycle")
                try:
                    opportunities = self.scouting_bot.scan_and_filter_opportunities()
                    self.logger.info(f"Found {len(opportunities)} opportunities")
                    return []
                except Exception as e:
                    self.logger.error(f"Minimal cycle failed: {e}")
                    return []
            
            def get_system_health(self):
                return {
                    'status': 'minimal',
                    'timestamp': datetime.now().isoformat(),
                    'message': 'Running in minimal mode'
                }
            
            def cleanup_old_conversations(self):
                pass
        
        return MinimalOrchestrator(self.scouting_bot, self.bingx_api)
    
    def graceful_shutdown(self, signum, frame):
        self.logger.info(f"Shutdown signal received (signal {signum}). Stopping trading system...")
        self.is_running = False
    
    async def run(self):
        """Main trading system loop"""
        self.is_running = True
        self.logger.info("Starting Advanced Trading System main loop")
        
        # Initial health check
        health_status = self.orchestrator.get_system_health()
        self.logger.info(f"Initial system health: {health_status['status']}")
        
        cycle_interval = getattr(self.config.app, 'cycle_interval_seconds', 300)
        error_count = 0
        max_consecutive_errors = 5
        
        while self.is_running:
            cycle_start_time = datetime.now()
            
            try:
                # Run trading cycle
                self.logger.debug("Starting trading cycle")
                executed_trades = self.orchestrator.run_complete_trading_cycle()
                
                # Update monitoring metrics if available
                if self.monitor:
                    try:
                        positions = self.bingx_api.fetch_open_positions()
                        self.monitor.update_business_metrics(
                            active_positions=len(positions),
                            portfolio_value=100000,  # Placeholder - would calculate from positions
                            daily_pnl=0  # Placeholder - would calculate from positions
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to update monitoring metrics: {e}")
                
                # Log cycle completion
                cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
                self.logger.info(
                    f"Trading cycle completed in {cycle_duration:.2f}s. "
                    f"Executed {len(executed_trades)} trades."
                )
                
                # Reset error count on successful cycle
                error_count = 0
                
                # Periodic maintenance tasks
                if int(cycle_start_time.timestamp()) % 3600 == 0:  # Every hour
                    self.logger.info("Running periodic maintenance")
                    try:
                        if hasattr(self.orchestrator, 'cleanup_old_conversations'):
                            self.orchestrator.cleanup_old_conversations()
                    except Exception as e:
                        self.logger.warning(f"Maintenance task failed: {e}")
                
                # Wait for next cycle
                await asyncio.sleep(cycle_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                error_count += 1
                self.logger.error(f"Error in trading cycle (#{error_count}): {e}")
                
                # Implement exponential backoff for errors
                if error_count >= max_consecutive_errors:
                    self.logger.critical(
                        f"Too many consecutive errors ({error_count}). "
                        "Stopping system to prevent damage."
                    )
                    break
                
                # Wait longer after errors, with exponential backoff
                error_delay = min(300, 30 * (2 ** (error_count - 1)))  # Max 5 minutes
                self.logger.info(f"Waiting {error_delay}s before retry due to errors")
                await asyncio.sleep(error_delay)
        
        self.logger.info("Trading system main loop stopped")
    
    async def run_health_checks(self):
        """Run periodic health checks"""
        if not self.monitor:
            self.logger.debug("Health checks disabled - monitoring not available")
            return
        
        health_check_interval = 300  # 5 minutes
        
        while self.is_running:
            try:
                self.logger.debug("Running health checks")
                
                # Check orchestrator health
                health_status = self.orchestrator.get_system_health()
                
                # Update component health in monitor
                if 'components' in health_status:
                    for component, status in health_status['components'].items():
                        health_level = 'healthy'
                        error_msg = None
                        
                        if isinstance(status, str):
                            if 'error' in status.lower():
                                health_level = 'unhealthy'
                                error_msg = status
                            elif 'degraded' in status.lower():
                                health_level = 'degraded'
                        
                        self.monitor.update_component_health(
                            component, 
                            health_level,
                            error=error_msg
                        )
                
                await asyncio.sleep(health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                await asyncio.sleep(60)  # Shorter retry for health checks
    
    async def run_performance_monitoring(self):
        """Run performance monitoring and alerting"""
        if not self.monitor:
            return
        
        monitoring_interval = 60  # 1 minute
        
        while self.is_running:
            try:
                # Monitor system performance
                # This could include CPU, memory, disk usage, etc.
                # For now, we'll just log periodic status
                
                overall_health = {}
                if hasattr(self.monitor, 'get_overall_health'):
                    overall_health = self.monitor.get_overall_health()
                
                if overall_health.get('status') == 'unhealthy':
                    self.logger.warning(
                        f"System health degraded: {overall_health.get('message', 'Unknown issue')}"
                    )
                
                await asyncio.sleep(monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Performance monitoring failed: {e}")
                await asyncio.sleep(60)

def setup_environment():
    """Setup environment variables if not already set"""
    default_env_vars = {
        'IA_STRATEGY_URL': 'http://localhost:8001/api/analyze',
        'IA_SPECIALIZED_URL': 'http://localhost:8002/api/consult', 
        'BINGX_IA_URL': 'http://localhost:8003/api/position',
        'MAX_RETRY_ATTEMPTS': '3',
        'REQUEST_TIMEOUT': '30',
        'MAX_POSITION_SIZE': '0.1',
        'RISK_PER_TRADE': '0.02',
        'CYCLE_INTERVAL': '300',
        'LOG_LEVEL': 'INFO',
        'ENVIRONMENT': 'development'
    }
    
    for key, default_value in default_env_vars.items():
        if key not in os.environ:
            os.environ[key] = default_value
            logging.debug(f"Set default environment variable: {key}={default_value}")

async def main():
    """Main application entry point"""
    
    # Setup environment
    setup_environment()
    
    # Parse command line arguments
    use_mocks = '--production' not in sys.argv
    if use_mocks:
        print("Running in MOCK mode for testing and development")
        print("Use --production flag for production mode (requires real API implementations)")
    
    try:
        # Create and configure system
        system = TradingSystem(use_mocks=use_mocks)
        
        # Create list of concurrent tasks
        tasks = [system.run()]
        
        # Add optional tasks if monitoring is available
        if system.monitor:
            tasks.extend([
                system.run_health_checks(),
                system.run_performance_monitoring()
            ])
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
        
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
    except Exception as e:
        logging.error(f"Application failed with error: {e}")
        raise
    finally:
        logging.info("Application shutdown complete")

def export_env_template():
    """Export environment template file"""
    template_content = '''# Advanced Trading System Environment Configuration

# Core Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
CYCLE_INTERVAL=300

# AI Service URLs
IA_STRATEGY_URL=http://localhost:8001/api/analyze
IA_SPECIALIZED_URL=http://localhost:8002/api/consult
BINGX_IA_URL=http://localhost:8003/api/position

# API Configuration
MAX_RETRY_ATTEMPTS=3
REQUEST_TIMEOUT=30

# Trading Parameters
MAX_POSITION_SIZE=0.1
RISK_PER_TRADE=0.02

# Security Settings (if using real APIs)
# API_KEY=your_api_key_here
# API_SECRET=your_secret_here
# WEBHOOK_SECRET=your_webhook_secret_here

# Monitoring Configuration
ENABLE_PROMETHEUS=false
PROMETHEUS_PORT=8000
HEALTH_PORT=8080

# Optional: Database Configuration
# DATABASE_PATH=trading_system.db
# DB_BACKUP_INTERVAL=24

# Optional: Alert Configuration
# ALERT_WEBHOOK_URL=https://your-webhook-url.com
'''
    
    with open('.env.template', 'w') as f:
        f.write(template_content)
    print("Environment template exported to .env.template")

if __name__ == "__main__":
    # Handle special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == "--export-env":
            export_env_template()
            sys.exit(0)
        elif sys.argv[1] == "--help":
            print("""
Advanced Trading System

Usage:
  python main_fixed.py [options]

Options:
  --production     Run in production mode (requires real API implementations)
  --export-env     Export environment variable template
  --help          Show this help message

Default behavior:
  - Runs in mock/development mode
  - Uses simulated APIs and data
  - Safe for testing and development

Environment Setup:
  1. Copy .env.template to .env (use --export-env to generate template)
  2. Modify .env with your settings
  3. Install dependencies: pip install -r requirements.txt (if available)

Required Python packages:
  - requests (critical)
  - Optional: tenacity, cachetools, prometheus-client, flask
            """)
            sys.exit(0)
    
    # Run the application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)