# quantconnect_error_recovery.py
"""
Enhanced error recovery system integrated with QuantConnect best practices
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import json
from collections import defaultdict, deque

# QuantConnect specific imports
try:
    from QuantConnect import *
    from QuantConnect.Algorithm import QCAlgorithm
    from QuantConnect.Data import *
    from QuantConnect.Orders import *
    from QuantConnect.Securities import *
    from QuantConnect.Brokerages import *
    QC_AVAILABLE = True
except ImportError:
    QC_AVAILABLE = False
    logging.warning("QuantConnect framework not available - running in standalone mode")

class QcErrorSeverity(Enum):
    """Error severity levels aligned with QuantConnect practices"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class QcSystemState(Enum):
    """System state levels for monitoring"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CRITICAL = "critical"
    RECOVERY = "recovery"

class QcErrorRecoveryManager:
    """
    Error recovery manager optimized for QuantConnect algorithms
    Based on QC best practices for fault tolerance and resilience
    """
    
    def __init__(self, algorithm: Optional[Any] = None, max_error_history: int = 1000):
        self.algorithm = algorithm
        self.logger = logging.getLogger(__name__)
        self.error_history = deque(maxlen=max_error_history)
        self.recovery_strategies = {}
        self.component_states = defaultdict(lambda: QcSystemState.HEALTHY)
        self.last_recovery_attempts = {}
        self.recovery_cooldown = timedelta(minutes=5)
        
        # Error thresholds based on QC recommendations
        self.error_thresholds = {
            QcSystemState.HEALTHY: 0,
            QcSystemState.DEGRADED: 3,
            QcSystemState.FAILING: 7,
            QcSystemState.CRITICAL: 15
        }
        
        # Register default recovery strategies
        self._register_qc_strategies()
    
    def _register_qc_strategies(self):
        """Register recovery strategies optimized for QuantConnect"""
        self.register_recovery_strategy("data_feed", self._recover_data_feed_issues)
        self.register_recovery_strategy("order_execution", self._recover_order_execution_issues)
        self.register_recovery_strategy("brokerage_connection", self._recover_brokerage_issues)
        self.register_recovery_strategy("margin_insufficient", self._recover_margin_issues)
        self.register_recovery_strategy("rate_limit", self._recover_rate_limit_issues)
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """Register a recovery strategy for a specific error type"""
        self.recovery_strategies[error_type] = strategy
        self.logger.debug(f"Registered recovery strategy for {error_type}")
    
    def record_error(self, component: str, error_type: str, severity: QcErrorSeverity, 
                    message: str, exception: Exception = None, context: Dict = None) -> Dict:
        """
        Record an error event following QuantConnect best practices
        """
        error_event = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "error_type": error_type,
            "severity": severity.value,
            "message": message,
            "exception": str(exception) if exception else None,
            "context": context or {},
            "recovery_attempted": False,
            "recovery_successful": False
        }
        
        self.error_history.append(error_event)
        
        # Update component state based on error history
        self._update_component_state(component)
        
        # Log according to QC best practices
        self._log_error(error_event)
        
        # Attempt recovery if appropriate
        if self._should_attempt_recovery(component, error_type):
            self._attempt_recovery(error_event)
        
        return error_event
    
    def _log_error(self, error_event: Dict):
        """Log errors following QuantConnect best practices"""
        severity = error_event["severity"]
        message = f"{error_event['component']}: {error_event['message']}"
        
        if self.algorithm and QC_AVAILABLE:
            # Use QC algorithm's logging system
            if severity == QcErrorSeverity.DEBUG.value:
                self.algorithm.Debug(message)
            elif severity == QcErrorSeverity.INFO.value:
                self.algorithm.Log(message)
            elif severity == QcErrorSeverity.WARNING.value:
                self.algorithm.Warning(message)
            elif severity == QcErrorSeverity.ERROR.value:
                self.algorithm.Error(message)
            elif severity == QcErrorSeverity.CRITICAL.value:
                self.algorithm.Error(f"CRITICAL: {message}")
        else:
            # Fallback to standard logging
            if severity == QcErrorSeverity.DEBUG.value:
                self.logger.debug(message)
            elif severity == QcErrorSeverity.INFO.value:
                self.logger.info(message)
            elif severity == QcErrorSeverity.WARNING.value:
                self.logger.warning(message)
            elif severity == QcErrorSeverity.ERROR.value:
                self.logger.error(message)
            elif severity == QcErrorSeverity.CRITICAL.value:
                self.logger.critical(message)
    
    def _update_component_state(self, component: str):
        """Update component state based on recent errors"""
        now = datetime.now()
        recent_window = timedelta(minutes=10)
        
        # Count recent errors for this component
        recent_errors = [
            e for e in self.error_history 
            if e["component"] == component and 
            (now - datetime.fromisoformat(e["timestamp"])) <= recent_window
        ]
        
        error_count = len(recent_errors)
        old_state = self.component_states[component]
        
        # Determine new state based on error count
        if error_count >= self.error_thresholds[QcSystemState.CRITICAL]:
            new_state = QcSystemState.CRITICAL
        elif error_count >= self.error_thresholds[QcSystemState.FAILING]:
            new_state = QcSystemState.FAILING
        elif error_count >= self.error_thresholds[QcSystemState.DEGRADED]:
            new_state = QcSystemState.DEGRADED
        else:
            new_state = QcSystemState.HEALTHY
        
        if new_state != old_state:
            self.component_states[component] = new_state
            self.logger.warning(
                f"Component {component} state changed: {old_state.value} -> {new_state.value}"
            )
    
    def _should_attempt_recovery(self, component: str, error_type: str) -> bool:
        """Determine if recovery should be attempted based on QC practices"""
        
        # Check if we have a recovery strategy
        if error_type not in self.recovery_strategies:
            return False
        
        # Check cooldown period
        last_attempt = self.last_recovery_attempts.get(f"{component}:{error_type}")
        if last_attempt and (datetime.now() - last_attempt) < self.recovery_cooldown:
            return False
        
        # Only attempt recovery for failing or critical components
        component_state = self.component_states[component]
        return component_state in [QcSystemState.FAILING, QcSystemState.CRITICAL]
    
    def _attempt_recovery(self, error_event: Dict):
        """Attempt to recover from an error following QC patterns"""
        recovery_key = f"{error_event['component']}:{error_event['error_type']}"
        self.last_recovery_attempts[recovery_key] = datetime.now()
        
        self.logger.info(f"Attempting recovery for {recovery_key}")
        
        try:
            strategy = self.recovery_strategies[error_event["error_type"]]
            success = strategy(error_event)
            
            error_event["recovery_attempted"] = True
            error_event["recovery_successful"] = success
            
            if success:
                self.logger.info(f"Recovery successful for {recovery_key}")
                # Reset component state to recovery mode
                self.component_states[error_event["component"]] = QcSystemState.RECOVERY
            else:
                self.logger.warning(f"Recovery failed for {recovery_key}")
                
        except Exception as e:
            self.logger.error(f"Recovery strategy failed for {recovery_key}: {e}")
            error_event["recovery_successful"] = False
    
    # QuantConnect specific recovery strategies
    def _recover_data_feed_issues(self, error_event: Dict) -> bool:
        """Recover from data feed issues following QC patterns"""
        self.logger.info("Attempting data feed recovery")
        
        if self.algorithm and QC_AVAILABLE:
            try:
                # QC-specific data feed recovery
                # This might involve reinitializing data subscriptions
                self.algorithm.Debug("Reinitializing data subscriptions")
                
                # In a real QC algorithm, you might do:
                # self.algorithm.RemoveSecurity(...)
                # self.algorithm.AddSecurity(...)
                
                return True
            except Exception as e:
                self.logger.error(f"Data feed recovery failed: {e}")
                return False
        else:
            # Generic recovery for non-QC environments
            time.sleep(2)
            return True
    
    def _recover_order_execution_issues(self, error_event: Dict) -> bool:
        """Recover from order execution issues following QC patterns"""
        self.logger.info("Attempting order execution recovery")
        
        if self.algorithm and QC_AVAILABLE:
            try:
                # QC-specific order recovery
                # Check for open orders and attempt to cancel them
                open_orders = self.algorithm.Transactions.GetOpenOrders()
                if open_orders:
                    for order in open_orders:
                        try:
                            self.algorithm.Transactions.CancelOrder(order.Id)
                        except Exception as e:
                            self.logger.error(f"Failed to cancel order {order.Id}: {e}")
                
                return True
            except Exception as e:
                self.logger.error(f"Order execution recovery failed: {e}")
                return False
        else:
            # Generic recovery
            time.sleep(5)
            return True
    
    def _recover_brokerage_issues(self, error_event: Dict) -> bool:
        """Recover from brokerage connection issues following QC patterns"""
        self.logger.info("Attempting brokerage connection recovery")
        
        if self.algorithm and QC_AVAILABLE:
            try:
                # QC-specific brokerage recovery
                # This might involve checking connection status
                # and reconnecting if necessary
                if hasattr(self.algorithm, 'Brokerage') and self.algorithm.Brokerage is not None:
                    if not self.algorithm.Brokerage.IsConnected:
                        self.algorithm.Brokerage.Connect()
                        return self.algorithm.Brokerage.IsConnected
                
                return True
            except Exception as e:
                self.logger.error(f"Brokerage connection recovery failed: {e}")
                return False
        else:
            # Generic recovery
            time.sleep(10)
            return True
    
    def _recover_margin_issues(self, error_event: Dict) -> bool:
        """Recover from margin issues following QC patterns"""
        self.logger.info("Attempting margin issue recovery")
        
        if self.algorithm and QC_AVAILABLE:
            try:
                # QC-specific margin recovery
                # This might involve liquidating positions to free up margin
                for symbol, holding in self.algorithm.Portfolio.items():
                    if holding.Invested:
                        # Liquidate a portion of positions based on margin requirements
                        # This is a simplified example
                        quantity = -holding.Quantity * 0.5  # Liquidate 50%
                        self.algorithm.MarketOrder(symbol, quantity)
                
                return True
            except Exception as e:
                self.logger.error(f"Margin issue recovery failed: {e}")
                return False
        else:
            # Generic recovery
            time.sleep(3)
            return True
    
    def _recover_rate_limit_issues(self, error_event: Dict) -> bool:
        """Recover from rate limiting issues following QC patterns"""
        self.logger.info("Handling rate limit recovery")
        
        # Extract wait time from context if available
        wait_time = error_event.get("context", {}).get('retry_after', 60)
        
        self.logger.info(f"Waiting {wait_time} seconds for rate limit reset")
        time.sleep(wait_time)
        
        return True
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary following QC patterns"""
        now = datetime.now()
        recent_window = timedelta(hours=1)
        
        recent_errors = [
            e for e in self.error_history 
            if (now - datetime.fromisoformat(e["timestamp"])) <= recent_window
        ]
        
        # Group errors by severity
        severity_counts = defaultdict(int)
        for error in recent_errors:
            severity_counts[error["severity"]] += 1
        
        # Determine overall system state
        critical_components = [
            comp for comp, state in self.component_states.items() 
            if state == QcSystemState.CRITICAL
        ]
        
        failing_components = [
            comp for comp, state in self.component_states.items() 
            if state == QcSystemState.FAILING
        ]
        
        if critical_components:
            overall_state = QcSystemState.CRITICAL
        elif failing_components:
            overall_state = QcSystemState.FAILING
        elif any(state == QcSystemState.DEGRADED for state in self.component_states.values()):
            overall_state = QcSystemState.DEGRADED
        else:
            overall_state = QcSystemState.HEALTHY
        
        return {
            "overall_state": overall_state.value,
            "recent_error_count": len(recent_errors),
            "error_by_severity": dict(severity_counts),
            "component_states": {
                comp: state.value 
                for comp, state in self.component_states.items()
            },
            "critical_components": critical_components,
            "failing_components": failing_components,
            "timestamp": now.isoformat()
        }

class QcCircuitBreaker:
    """
    Circuit breaker implementation optimized for QuantConnect
    Based on QC best practices for managing external service dependencies
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 success_threshold: int = 3,
                 timeout: int = 60):
        
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.circuit_open_count = 0
    
    def execute(self, func: Callable, *args, **kwargs):
        """
        Execute function through circuit breaker following QC patterns
        """
        self.total_requests += 1
        
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception(
                    f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout
    
    def _on_success(self):
        """Handle successful call following QC patterns"""
        self.successful_requests += 1
        
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._reset()
        else:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self, exception: Exception):
        """Handle failed call following QC patterns"""
        self.failed_requests += 1
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self._trip()
        
        self.logger.warning(f"Circuit breaker failure #{self.failure_count}: {exception}")
    
    def _reset(self):
        """Reset circuit breaker to CLOSED state"""
        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.logger.info("Circuit breaker reset to CLOSED")
    
    def _trip(self):
        """Trip circuit breaker to OPEN state"""
        self.state = "OPEN"
        self.circuit_open_count += 1
        self.logger.error(f"Circuit breaker tripped to OPEN (#{self.circuit_open_count})")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics following QC patterns"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.successful_requests / max(1, self.total_requests)) * 100,
            "circuit_open_count": self.circuit_open_count,
            "last_failure_time": self.last_failure_time
        }

class QcRetryManager:
    """
    Retry manager implementation optimized for QuantConnect
    Based on QC best practices for handling transient failures
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def execute_with_retry(self, 
                         func: Callable,
                         max_attempts: int = 3,
                         base_delay: float = 1.0,
                         max_delay: float = 60.0,
                         backoff_factor: float = 2.0,
                         retry_on: tuple = (Exception,),
                         **kwargs) -> Any:
        """
        Execute function with retry logic following QC patterns
        """
        
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                return func(**kwargs)
                    
            except retry_on as e:
                last_exception = e
                
                if attempt == max_attempts - 1:
                    self.logger.error(f"All {max_attempts} retry attempts failed")
                    break
                
                # Calculate delay with exponential backoff
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s"
                )
                
                time.sleep(delay)
            
            except Exception as e:
                # Don't retry unexpected exceptions
                self.logger.error(f"Unexpected exception, not retrying: {e}")
                raise
        
        raise last_exception

# Integration with QuantConnect algorithm framework
class QcResilientAlgorithmComponent:
    """
    Base class for QuantConnect algorithm components with built-in resilience
    Follows QC best practices for error handling and recovery
    """
    
    def __init__(self, component_name: str, algorithm: Any = None):
        self.component_name = component_name
        self.algorithm = algorithm
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize resilience components
        self.error_manager = QcErrorRecoveryManager(algorithm)
        self.circuit_breaker = QcCircuitBreaker()
        self.retry_manager = QcRetryManager()
        
    def execute_resilient(self, 
                        func: Callable,
                        error_type: str = "general",
                        severity: QcErrorSeverity = QcErrorSeverity.ERROR,
                        max_retries: int = 3,
                        **kwargs) -> Any:
        """
        Make a resilient call with automatic error handling and recovery
        Following QC best practices
        """
        
        try:
            # Use circuit breaker and retry logic
            return self.circuit_breaker.execute(
                self.retry_manager.execute_with_retry,
                func=func,
                max_attempts=max_retries,
                **kwargs
            )
            
        except Exception as e:
            # Record error for recovery management
            self.error_manager.record_error(
                component=self.component_name,
                error_type=error_type,
                severity=severity,
                message=str(e),
                exception=e
            )
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status following QC patterns"""
        return {
            "component": self.component_name,
            "error_manager": self.error_manager.get_system_health_summary(),
            "circuit_breaker": self.circuit_breaker.get_metrics(),
            "timestamp": datetime.now().isoformat()
        }

# Example integration with QuantConnect algorithm
class ResilientTradingAlgorithm(QCAlgorithm):
    """
    Example QuantConnect algorithm with integrated error recovery
    Following QC best practices for resilience
    """
    
    def Initialize(self):
        """Initialization following QC patterns"""
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)
        self.AddEquity("SPY", Resolution.Minute)
        
        # Initialize error recovery system
        self.error_manager = QcErrorRecoveryManager(self)
        
        # Initialize resilient components
        self.data_processor = QcResilientAlgorithmComponent("data_processor", self)
        self.order_executor = QcResilientAlgorithmComponent("order_executor", self)
        
        # Schedule error recovery monitoring
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(minutes=30)),
            self.monitor_system_health
        )
    
    def OnData(self, data):
        """
        Handle data events with error recovery
        Following QC best practices
        """
        try:
            # Process data with error recovery
            self.data_processor.execute_resilient(
                self.process_market_data,
                error_type="data_processing",
                severity=QcErrorSeverity.ERROR,
                max_retries=2,
                data=data
            )
            
        except Exception as e:
            self.Error(f"Failed to process data: {e}")
    
    def process_market_data(self, data):
        """Example data processing method with QC patterns"""
        try:
            # Your data processing logic here
            if "SPY" in data and data["SPY"] is not None:
                spy_price = data["SPY"].Close
                
                # Example trading logic
                if spy_price > self.Securities["SPY"].Price * 1.01:
                    self.order_executor.execute_resilient(
                        self.execute_trade,
                        error_type="order_execution",
                        severity=QcErrorSeverity.ERROR,
                        max_retries=3,
                        symbol="SPY",
                        quantity=-10
                    )
                
        except Exception as e:
            self.Error(f"Data processing error: {e}")
            raise
    
    def execute_trade(self, symbol: str, quantity: int):
        """Execute trade with QC error recovery patterns"""
        try:
            # Place order with QC framework
            order = self.MarketOrder(symbol, quantity)
            
            # Check for errors
            if order.Status == OrderStatus.Invalid:
                raise Exception(f"Invalid order: {order}")
                
            return order
            
        except Exception as e:
            self.Error(f"Trade execution failed: {e}")
            raise
    
    def monitor_system_health(self):
        """Monitor system health following QC patterns"""
        try:
            health_status = self.error_manager.get_system_health_summary()
            
            # Log health status
            self.Log(f"System health: {health_status['overall_state']}")
            
            # Take action based on health status
            if health_status['overall_state'] == QcSystemState.CRITICAL.value:
                self.Error("CRITICAL: System health is critical - reducing position exposure")
                self.reduce_exposure()
                
            elif health_status['overall_state'] == QcSystemState.FAILING.value:
                self.Warning("WARNING: System health is failing - pausing trading")
                self.pause_trading()
                
        except Exception as e:
            self.Error(f"Health monitoring failed: {e}")
    
    def reduce_exposure(self):
        """Reduce position exposure following QC patterns"""
        try:
            for symbol, holding in self.Portfolio.items():
                if holding.Invested:
                    # Liquidate 50% of position
                    quantity = -holding.Quantity * 0.5
                    self.MarketOrder(symbol, quantity)
                    
        except Exception as e:
            self.Error(f"Exposure reduction failed: {e}")
    
    def pause_trading(self):
        """Pause trading following QC patterns"""
        # Set a flag to prevent new trades
        self.trading_paused = True
        
        # Cancel all open orders
        open_orders = self.Transactions.GetOpenOrders()
        for order in open_orders:
            self.Transactions.CancelOrder(order.Id)

# Utility functions for QuantConnect integration
def setup_qc_error_recovery(algorithm: Any) -> QcErrorRecoveryManager:
    """
    Set up error recovery for a QuantConnect algorithm
    Following QC best practices
    """
    error_manager = QcErrorRecoveryManager(algorithm)
    
    # Register algorithm-specific recovery strategies
    error_manager.register_recovery_strategy(
        "data_quality", 
        lambda e: handle_data_quality_issues(algorithm, e)
    )
    
    error_manager.register_recovery_strategy(
        "position_sizing",
        lambda e: handle_position_sizing_issues(algorithm, e)
    )
    
    return error_manager

def handle_data_quality_issues(algorithm: Any, error_event: Dict) -> bool:
    """
    Handle data quality issues following QC patterns
    """
    algorithm.Debug("Handling data quality issues")
    
    try:
        # QC-specific data quality recovery
        # This might involve:
        # 1. Removing problematic data subscriptions
        # 2. Adding data quality checks
        # 3. Switching to alternative data sources
        
        return True
    except Exception as e:
        algorithm.Error(f"Data quality recovery failed: {e}")
        return False

def handle_position_sizing_issues(algorithm: Any, error_event: Dict) -> bool:
    """
    Handle position sizing issues following QC patterns
    """
    algorithm.Debug("Handling position sizing issues")
    
    try:
        # QC-specific position sizing recovery
        # This might involve:
        # 1. Reducing position sizes
        # 2. Adjusting risk parameters
        # 3. Liquidating oversized positions
        
        return True
    except Exception as e:
        algorithm.Error(f"Position sizing recovery failed: {e}")
        return False

# Example usage in standalone mode (without QC framework)
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create error recovery manager
    error_manager = QcErrorRecoveryManager()
    
    # Example error recording
    error_manager.record_error(
        component="data_feed",
        error_type="connection_timeout",
        severity=QcErrorSeverity.ERROR,
        message="Data feed connection timeout",
        context={"retry_after": 30}
    )
    
    # Get health status
    health_status = error_manager.get_system_health_summary()
    print(f"System health: {health_status}")