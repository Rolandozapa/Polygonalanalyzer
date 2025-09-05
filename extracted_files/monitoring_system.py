import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import requests

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
    from flask import Flask, jsonify, request
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logging.warning("Monitoring features disabled. Install: pip install prometheus-client flask")

@dataclass
class HealthStatus:
    """Health status for system components"""
    component: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    additional_info: Optional[Dict] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics for system monitoring"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_reset: datetime = None
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def failure_rate(self) -> float:
        return 100 - self.success_rate

class SystemMonitor:
    """Comprehensive system monitoring"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Health status tracking
        self.component_health: Dict[str, HealthStatus] = {}
        self.performance_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        
        # Alert tracking
        self.alerts = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []
        
        # Prometheus metrics if available
        if MONITORING_AVAILABLE and config and config.monitoring.enable_prometheus:
            self._init_prometheus_metrics()
            self.prometheus_enabled = True
        else:
            self.prometheus_enabled = False
        
        # Health check server
        self.health_app = Flask(__name__)
        self.health_server_thread = None
        self._setup_health_endpoints()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        self.registry = CollectorRegistry()
        
        self.request_count = Counter(
            'trading_requests_total',
            'Total number of requests',
            ['service', 'method', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'trading_request_duration_seconds',
            'Request duration in seconds',
            ['service', 'method'],
            registry=self.registry
        )
        
        self.component_health_gauge = Gauge(
            'trading_component_health',
            'Component health status (1=healthy, 0.5=degraded, 0=unhealthy)',
            ['component'],
            registry=self.registry
        )
        
        self.active_positions_gauge = Gauge(
            'trading_active_positions',
            'Number of active trading positions',
            registry=self.registry
        )
        
        self.portfolio_value_gauge = Gauge(
            'trading_portfolio_value_usd',
            'Current portfolio value in USD',
            registry=self.registry
        )
        
        self.daily_pnl_gauge = Gauge(
            'trading_daily_pnl_usd',
            'Daily profit/loss in USD',
            registry=self.registry
        )
    
    def _setup_health_endpoints(self):
        """Setup health check endpoints"""
        
        @self.health_app.route('/health')
        def health_check():
            """Main health check endpoint"""
            overall_status = self.get_overall_health()
            
            response = {
                'status': overall_status['status'],
                'timestamp': datetime.now().isoformat(),
                'components': {name: asdict(health) for name, health in self.component_health.items()},
                'metrics': self._get_summary_metrics()
            }
            
            status_code = 200 if overall_status['status'] == 'healthy' else 503
            return jsonify(response), status_code
        
        @self.health_app.route('/health/<component>')
        def component_health(component):
            """Individual component health"""
            if component in self.component_health:
                health = self.component_health[component]
                return jsonify(asdict(health))
            else:
                return jsonify({'error': f'Component {component} not found'}), 404
        
        @self.health_app.route('/metrics')
        def metrics_summary():
            """Metrics summary endpoint"""
            return jsonify(self._get_detailed_metrics())
        
        @self.health_app.route('/alerts')
        def recent_alerts():
            """Recent alerts endpoint"""
            return jsonify({
                'alerts': [asdict(alert) for alert in list(self.alerts)[-50:]]  # Last 50 alerts
            })
    
    def start_health_server(self, port: int = 8080):
        """Start health check server"""
        if self.health_server_thread is None:
            def run_server():
                self.health_app.run(host='0.0.0.0', port=port, debug=False)
            
            self.health_server_thread = threading.Thread(target=run_server, daemon=True)
            self.health_server_thread.start()
            self.logger.info(f"Health check server started on port {port}")
    
    def start_prometheus_server(self, port: int = 8000):
        """Start Prometheus metrics server"""
        if self.prometheus_enabled:
            start_http_server(port, registry=self.registry)
            self.logger.info(f"Prometheus metrics server started on port {port}")
    
    def record_request(self, service: str, method: str, duration: float, success: bool):
        """Record request metrics"""
        
        # Update performance metrics
        metrics = self.performance_metrics[service]
        metrics.total_requests += 1
        
        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
        
        # Update average response time (rolling average)
        if metrics.total_requests == 1:
            metrics.average_response_time = duration
        else:
            # Simple moving average
            weight = 0.1  # Weight for new measurement
            metrics.average_response_time = (
                (1 - weight) * metrics.average_response_time + 
                weight * duration
            )
        
        # Prometheus metrics
        if self.prometheus_enabled:
            status = 'success' if success else 'error'
            self.request_count.labels(service=service, method=method, status=status).inc()
            self.request_duration.labels(service=service, method=method).observe(duration)
    
    def update_component_health(self, component: str, status: str, 
                               response_time: float = None, error: str = None, 
                               additional_info: Dict = None):
        """Update health status of a component"""
        
        health_status = HealthStatus(
            component=component,
            status=status,
            last_check=datetime.now(),
            response_time=response_time,
            error_message=error,
            additional_info=additional_info or {}
        )
        
        self.component_health[component] = health_status
        
        # Update Prometheus gauge
        if self.prometheus_enabled:
            health_value = {'healthy': 1.0, 'degraded': 0.5, 'unhealthy': 0.0}.get(status, 0.0)
            self.component_health_gauge.labels(component=component).set(health_value)
        
        # Generate alert if status is not healthy
        if status != 'healthy':
            self._generate_alert(
                level='warning' if status == 'degraded' else 'critical',
                component=component,
                message=f"Component {component} status: {status}",
                details={'error': error, 'response_time': response_time}
            )
    
    def check_component_health(self, component: str, check_func: Callable) -> HealthStatus:
        """Check health of a component using provided function"""
        start_time = time.time()
        
        try:
            result = check_func()
            response_time = time.time() - start_time
            
            if result.get('status') == 'healthy':
                self.update_component_health(
                    component, 'healthy', response_time, 
                    additional_info=result.get('info', {})
                )
            else:
                self.update_component_health(
                    component, 'degraded', response_time,
                    error=result.get('error', 'Unknown issue'),
                    additional_info=result.get('info', {})
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            self.update_component_health(
                component, 'unhealthy', response_time, 
                error=str(e)
            )
        
        return self.component_health[component]
    
    def get_overall_health(self) -> Dict:
        """Get overall system health status"""
        if not self.component_health:
            return {'status': 'unknown', 'message': 'No components monitored'}
        
        healthy_count = sum(1 for h in self.component_health.values() if h.status == 'healthy')
        degraded_count = sum(1 for h in self.component_health.values() if h.status == 'degraded')
        unhealthy_count = sum(1 for h in self.component_health.values() if h.status == 'unhealthy')
        
        total_components = len(self.component_health)
        
        if unhealthy_count > 0:
            status = 'unhealthy'
            message = f"{unhealthy_count} unhealthy components"
        elif degraded_count > 0:
            status = 'degraded'
            message = f"{degraded_count} degraded components"
        else:
            status = 'healthy'
            message = f"All {total_components} components healthy"
        
        return {
            'status': status,
            'message': message,
            'healthy': healthy_count,
            'degraded': degraded_count,
            'unhealthy': unhealthy_count,
            'total': total_components
        }
    
    def update_business_metrics(self, active_positions: int = None, 
                               portfolio_value: float = None, 
                               daily_pnl: float = None):
        """Update business-specific metrics"""
        if self.prometheus_enabled:
            if active_positions is not None:
                self.active_positions_gauge.set(active_positions)
            
            if portfolio_value is not None:
                self.portfolio_value_gauge.set(portfolio_value)
            
            if daily_pnl is not None:
                self.daily_pnl_gauge.set(daily_pnl)
    
    def _generate_alert(self, level: str, component: str, message: str, details: Dict = None):
        """Generate system alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'component': component,
            'message': message,
            'details': details or {}
        }
        
        self.alerts.append(alert)
        self.logger.log(
            logging.WARNING if level == 'warning' else logging.ERROR,
            f"ALERT [{level.upper()}] {component}: {message}"
        )
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def _get_summary_metrics(self) -> Dict:
        """Get summary metrics"""
        