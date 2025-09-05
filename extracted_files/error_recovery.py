# error_recovery.py
"""
Enhanced error recovery and resilience patterns for the trading system
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SystemState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CRITICAL = "critical"
    RECOVERY = "recovery"

@dataclass
class ErrorEvent:
    """Represents an error event in the system"""
    timestamp: datetime
    component: str
    error_type: str
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False

class ErrorRecoveryManager:
    """Manages error recovery strategies across the system"""
    
    def __init__(self, max_error_history: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.error_history = deque(maxlen=max_error_history)
        self.recovery_strategies: Dict[str, Callable] = {}
        self.component_states: Dict[str, SystemState] = defaultdict(lambda: SystemState.HEALTHY)
        self.last_recovery_attempts: Dict[str, datetime] = {}
        self.recovery_cooldown = timedelta(minutes=5)
        
        # Error thresholds for state transitions
        self.error_thresholds = {
            SystemState.HEALTHY: 0,
            SystemState.DEGRADED: 3,
            SystemState.FAILING: 7,
            SystemState.CRITICAL: 15
        }
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default recovery strategies"""
        self.register_recovery_strategy("network", self._recover_network_issues)
        self.register_recovery_strategy("database", self._recover_database_issues)
        self.register_recovery_strategy("ai_service", self._recover_ai_service_issues)
        self.register_recovery_strategy("api_limit", self._recover_rate_limit_issues)
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """Register a recovery strategy for a specific error type"""
        self.recovery_strategies[error_type] = strategy
        self.logger.debug(f"Registered recovery strategy for {error_type}")
    
    def record_error(self, component: str, error_type: str, severity: ErrorSeverity, 
                    message: str, exception: Exception = None, context: Dict = None) -> ErrorEvent:
        """Record an error event and trigger recovery if needed"""
        
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            component=component,
            error_type=error_type,
            severity=severity,
            message=message,
            exception=exception,
            context=context or {}
        )
        
        self.error_history.append(error_event)
        
        # Update component state based on error history
        self._update_component_state(component)
        
        # Attempt recovery if appropriate
        if self._should_attempt_recovery(component, error_type):
            asyncio.create_task(self._attempt_recovery(error_event))
        
        return error_event
    
    def _update_component_state(self, component: str):
        """Update component state based on recent errors"""
        now = datetime.now()
        recent_window = timedelta(minutes=10)
        
        # Count recent errors for this component
        recent_errors = [
            e for e in self.error_history 
            if e.component == component and (now - e.timestamp) <= recent_window
        ]
        
        error_count = len(recent_errors)
        old_state = self.component_states[component]
        
        # Determine new state based on error count
        if error_count >= self.error_thresholds[SystemState.CRITICAL]:
            new_state = SystemState.CRITICAL
        elif error_count >= self.error_thresholds[SystemState.FAILING]:
            new_state = SystemState.FAILING
        elif error_count >= self.error_thresholds[SystemState.DEGRADED]:
            new_state = SystemState.DEGRADED
        else:
            new_state = SystemState.HEALTHY
        
        if new_state != old_state:
            self.component_states[component] = new_state
            self.logger.warning(
                f"Component {component} state changed: {old_state.value} -> {new_state.value}"
            )
    
    def _should_attempt_recovery(self, component: str, error_type: str) -> bool:
        """Determine if recovery should be attempted"""
        
        # Check if we have a recovery strategy
        if error_type not in self.recovery_strategies:
            return False
        
        # Check cooldown period
        last_attempt = self.last_recovery_attempts.get(f"{component}:{error_type}")
        if last_attempt and (datetime.now() - last_attempt) < self.recovery_cooldown:
            return False
        
        # Only attempt recovery for failing or critical components
        component_state = self.component_states[component]
        return component_state in [SystemState.FAILING, SystemState.CRITICAL]
    
    async def _attempt_recovery(self, error_event: ErrorEvent):
        """Attempt to recover from an error"""
        recovery_key = f"{error_event.component}:{error_event.error_type}"
        self.last_recovery_attempts[recovery_key] = datetime.now()
        
        self.logger.info(f"Attempting recovery for {recovery_key}")
        
        try:
            strategy = self.recovery_strategies[error_event.error_type]
            success = await strategy(error_event)
            
            error_event.recovery_attempted = True
            error_event.recovery_successful = success
            
            if success:
                self.logger.info(f"Recovery successful for {recovery_key}")
                # Reset component state to recovery mode
                self.component_states[error_event.component] = SystemState.RECOVERY
            else:
                self.logger.warning(f"Recovery failed for {recovery_key}")
                
        except Exception as e:
            self.logger.error(f"Recovery strategy failed for {recovery_key}: {e}")
            error_event.recovery_successful = False
    
    # Default recovery strategies
    async def _recover_network_issues(self, error_event: ErrorEvent) -> bool:
        """Recover from network connectivity issues"""
        self.logger.info("Attempting network recovery")
        
        # Wait and test connectivity
        await asyncio.sleep(2)
        
        try:
            import requests
            response = requests.get("https://httpbin.org/status/200", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def _recover_database_issues(self, error_event: ErrorEvent) -> bool:
        """Recover from database issues"""
        self.logger.info("Attempting database recovery")
        
        try:
            # Basic database connectivity test
            import sqlite3
            conn = sqlite3.connect(":memory:")
            conn.execute("SELECT 1")
            conn.close()
            return True
        except:
            return False
    
    async def _recover_ai_service_issues(self, error_event: ErrorEvent) -> bool:
        """Recover from AI service issues"""
        self.logger.info("Attempting AI service recovery")
        
        # Wait for service to potentially recover
        await asyncio.sleep(5)
        
        # Could implement service health check here
        return True  # Optimistically assume recovery
    
    async def _recover_rate_limit_issues(self, error_event: ErrorEvent) -> bool:
        """Recover from rate limiting issues"""
        self.logger.info("Handling rate limit recovery")
        
        # Extract wait time from context if available
        wait_time = error_event.context.get('retry_after', 60)
        
        self.logger.info(f"Waiting {wait_time} seconds for rate limit reset")
        await asyncio.sleep(wait_time)
        
        return True
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        now = datetime.now()
        recent_window = timedelta(hours=1)
        
        recent_errors = [
            e for e in self.error_history 
            if (now - e.timestamp) <= recent_window
        ]
        
        # Group errors by severity
        severity_counts = defaultdict(int)
        for error in recent_errors:
            severity_counts[error.severity.value] += 1
        
        # Determine overall system state
        critical_components = [
            comp for comp, state in self.component_states.items() 
            if state == SystemState.CRITICAL
        ]
        
        failing_components = [
            comp for comp, state in self.component_states.items() 
            if state == SystemState.FAILING
        ]
        
        if critical_components:
            overall_state = SystemState.CRITICAL
        elif failing_components:
            overall_state = SystemState.FAILING
        elif any(state == SystemState.DEGRADED for state in self.component_states.values()):
            overall_state = SystemState.DEGRADED
        else:
            overall_state = SystemState.HEALTHY
        
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

class CircuitBreakerAdvanced:
    """Advanced circuit breaker with more sophisticated logic"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 success_threshold: int = 3,
                 timeout: int = 60,
                 expected_exception: type = Exception):
        
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
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
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker"""
        self.total_requests += 1
        
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerOpenException(
                    f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                )
        
        try:
            # Add timeout to function call if it's async
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
            else:
                result = func(*args, **kwargs)
            
            self._on_success()
            return result
            
        except asyncio.TimeoutError as e:
            self._on_failure(e)
            raise
        except self.expected_exception as e:
            self._on_failure(e)
            raise
        except Exception as e:
            # Unexpected exceptions don't count as failures
            self.logger.warning(f"Unexpected exception in circuit breaker: {e}")
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.successful_requests += 1
        
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._reset()
        else:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self, exception: Exception):
        """Handle failed call"""
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
        """Get circuit breaker metrics"""
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

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class RetryManager:
    """Advanced retry manager with different strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def retry_with_backoff(self, 
                               func: Callable,
                               max_attempts: int = 3,
                               base_delay: float = 1.0,
                               max_delay: float = 60.0,
                               backoff_factor: float = 2.0,
                               jitter: bool = True,
                               retry_on: tuple = (Exception,),
                               **kwargs) -> Any:
        """Retry with exponential backoff"""
        
        import random
        
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(**kwargs)
                else:
                    return func(**kwargs)
                    
            except retry_on as e:
                last_exception = e
                
                if attempt == max_attempts - 1:
                    self.logger.error(f"All {max_attempts} retry attempts failed")
                    break
                
                # Calculate delay with exponential backoff
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                
                # Add jitter to prevent thundering herd
                if jitter:
                    delay = delay * (0.5 + random.random() * 0.5)
                
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s"
                )
                
                await asyncio.sleep(delay)
            
            except Exception as e:
                # Don't retry unexpected exceptions
                self.logger.error(f"Unexpected exception, not retrying: {e}")
                raise
        
        raise last_exception

# Integration with existing system
class ResilientTradingComponent:
    """Base class for trading components with built-in resilience"""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize resilience components
        self.error_manager = ErrorRecoveryManager()
        self.circuit_breaker = CircuitBreakerAdvanced()
        self.retry_manager = RetryManager()
        
    async def resilient_call(self, 
                           func: Callable,
                           error_type: str = "general",
                           severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                           max_retries: int = 3,
                           **kwargs) -> Any:
        """Make a resilient call with automatic error handling and recovery"""
        
        try:
            # Use circuit breaker and retry logic
            return await self.circuit_breaker.call(
                self.retry_manager.retry_with_backoff,
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
        """Get component health status"""
        return {
            "component": self.component_name,
            "error_manager": self.error_manager.get_system_health_summary(),
            "circuit_breaker": self.circuit_breaker.get_metrics(),
            "timestamp": datetime.now().isoformat()
        }

# Usage example
class ResilientAIClient(ResilientTradingComponent):
    """AI client with built-in resilience"""
    
    def __init__(self):
        super().__init__("ai_client")
    
    async def call_ai_service(self, url: str, payload: Dict) -> Dict:
        """Make resilient AI service call"""
        
        async def _make_request():
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=30) as response:
                    response.raise_for_status()
                    return await response.json()
        
        return await self.resilient_call(
            func=_make_request,
            error_type="ai_service",
            severity=ErrorSeverity.HIGH,
            max_retries=3
        )