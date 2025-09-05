import time
import hmac
import hashlib
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Callable, Any, Dict, List, Optional
from functools import wraps
import requests
from collections import defaultdict, deque

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

class RateLimiter:
    """Thread-safe rate limiter implementation"""
    
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = defaultdict(deque)
        self.logger = logging.getLogger(__name__)
    
    def is_allowed(self, key: str = "global") -> bool:
        """Check if a call is allowed for the given key"""
        now = time.time()
        call_times = self.calls[key]
        
        # Remove old calls outside the time window
        while call_times and call_times[0] <= now - self.time_window:
            call_times.popleft()
        
        # Check if we can make another call
        if len(call_times) < self.max_calls:
            call_times.append(now)
            return True
        
        return False
    
    def time_until_reset(self, key: str = "global") -> float:
        """Get time until the rate limit resets for the key"""
        now = time.time()
        call_times = self.calls[key]
        
        if not call_times or len(call_times) < self.max_calls:
            return 0.0
        
        oldest_call = call_times[0]
        return max(0, self.time_window - (now - oldest_call))

class CircuitBreaker:
    """Simple circuit breaker implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = logging.getLogger(__name__)
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker attempting reset (HALF_OPEN)")
            else:
                raise CircuitBreakerOpenException("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt a reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            self.logger.info("Circuit breaker reset to CLOSED")
    
    def _on_failure(self, exception: Exception):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.error(f"Circuit breaker opened due to {self.failure_count} failures")

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class RequestSigner:
    """Sign requests with HMAC for security"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret.encode('utf-8')
    
    def sign_request(self, method: str, endpoint: str, payload: Dict = None) -> Dict[str, str]:
        """Generate signature headers for request"""
        timestamp = str(int(time.time() * 1000))
        
        # Create string to sign
        if payload:
            body = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        else:
            body = ""
        
        string_to_sign = f"{method}{endpoint}{timestamp}{body}"
        
        # Generate signature
        signature = hmac.new(
            self.api_secret,
            string_to_sign.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            'X-API-Key': self.api_key,
            'X-Timestamp': timestamp,
            'X-Signature': signature,
            'Content-Type': 'application/json'
        }

class SecurityValidator:
    """Validate inputs and responses for security"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate trading symbol format"""
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Basic validation - adjust based on your exchange requirements
        if len(symbol) < 3 or len(symbol) > 20:
            return False
        
        # Only allow alphanumeric characters and dashes/underscores
        allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')
        if not set(symbol.upper()).issubset(allowed_chars):
            return False
        
        return True
    
    @staticmethod
    def validate_price(price: Any) -> bool:
        """Validate price value"""
        try:
            price_float = float(price)
            return price_float > 0 and price_float < 1e10  # Reasonable bounds
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_quantity(quantity: Any) -> bool:
        """Validate quantity value"""
        try:
            quantity_float = float(quantity)
            return 0 < quantity_float < 1e6  # Reasonable bounds
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def sanitize_log_data(data: Dict) -> Dict:
        """Remove sensitive information from log data"""
        sensitive_keys = {'api_key', 'api_secret', 'password', 'token', 'signature'}
        
        def _sanitize_recursive(obj):
            if isinstance(obj, dict):
                return {
                    key: "***REDACTED***" if key.lower() in sensitive_keys 
                    else _sanitize_recursive(value)
                    for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [_sanitize_recursive(item) for item in obj]
            else:
                return obj
        
        return _sanitize_recursive(data)

def secure_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Secure retry decorator with exponential backoff"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, ConnectionError, TimeoutError) as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        sleep_time = delay * (backoff ** attempt)
                        logging.warning(f"Attempt {attempt + 1} failed, retrying in {sleep_time:.2f}s: {e}")
                        time.sleep(sleep_time)
                    else:
                        logging.error(f"All {max_attempts} attempts failed for {func.__name__}")
                except Exception as e:
                    # Don't retry for unexpected exceptions
                    logging.error(f"Unexpected error in {func.__name__}: {e}")
                    raise
            
            raise last_exception
        return wrapper
    return decorator

def rate_limited(max_calls: int, time_window: int, key_func: Callable = None):
    """Rate limiting decorator"""
    rate_limiter = RateLimiter(max_calls, time_window)
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine rate limiting key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = "global"
            
            if not rate_limiter.is_allowed(key):
                wait_time = rate_limiter.time_until_reset(key)
                raise RateLimitExceeded(f"Rate limit exceeded. Try again in {wait_time:.2f} seconds")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    pass

class SecureAPIClient:
    """Secure API client with built-in protections"""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.signer = RequestSigner(api_key, api_secret)
        self.session = requests.Session()
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(max_calls=100, time_window=3600)  # 100 calls per hour
        self.logger = logging.getLogger(__name__)
        
        # Configure session with security headers
        self.session.headers.update({
            'User-Agent': 'Advanced-Trading-Bot/2.0',
            'Accept': 'application/json',
        })
    
    @secure_retry(max_attempts=3)
    def _make_request(self, method: str, endpoint: str, payload: Dict = None) -> requests.Response:
        """Make a secure HTTP request"""
        
        # Rate limiting check
        if not self.rate_limiter.is_allowed():
            wait_time = self.rate_limiter.time_until_reset()
            raise RateLimitExceeded(f"API rate limit exceeded. Wait {wait_time:.2f}s")
        
        # Sign the request
        headers = self.signer.sign_request(method, endpoint, payload)
        url = f"{self.base_url}{endpoint}"
        
        # Log request (sanitized)
        log_data = {
            'method': method,
            'url': url,
            'payload': SecurityValidator.sanitize_log_data(payload) if payload else None
        }
        self.logger.debug(f"Making API request: {log_data}")
        
        # Make request through circuit breaker
        def _request():
            if method.upper() == 'GET':
                return self.session.get(url, headers=headers, params=payload, timeout=30)
            else:
                return self.session.request(
                    method, url, headers=headers, 
                    json=payload, timeout=30
                )
        
        response = self.circuit_breaker.call(_request)
        
        # Validate response
        if not response.ok:
            self.logger.error(f"API request failed: {response.status_code} - {response.text}")
            response.raise_for_status()
        
        return response
    
    def get(self, endpoint: str, params: Dict = None) -> Dict:
        """Make secure GET request"""
        response = self._make_request('GET', endpoint, params)
        return response.json()
    
    def post(self, endpoint: str, payload: Dict = None) -> Dict:
        """Make secure POST request"""
        response = self._make_request('POST', endpoint, payload)
        return response.json()

class SecurityMonitor:
    """Monitor security events and anomalies"""
    
    def __init__(self):
        self.failed_requests = defaultdict(int)
        self.suspicious_activity = []
        self.logger = logging.getLogger(__name__)
    
    def log_failed_request(self, endpoint: str, reason: str):
        """Log failed request for monitoring"""
        self.failed_requests[endpoint] += 1
        
        if self.failed_requests[endpoint] > 10:  # Threshold
            self.logger.warning(f"High failure rate for endpoint {endpoint}: {reason}")
    
    def check_suspicious_activity(self, symbol: str, action: str, quantity: float):
        """Check for suspicious trading activity"""
        activity = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': action,
            'quantity': quantity
        }
        
        self.suspicious_activity.append(activity)
        
        # Keep only recent activity (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self.suspicious_activity = [
            a for a in self.suspicious_activity 
            if a['timestamp'] > cutoff
        ]
        
        # Check for suspicious patterns
        if len(self.suspicious_activity) > 50:  # Too many trades
            self.logger.warning("Suspicious activity: Too many trades in short period")
        
        # Check for large quantities
        if quantity > 100:  # Adjust threshold as needed
            self.logger.warning(f"Suspicious activity: Large quantity trade {quantity}")

def validate_ia_response(response: Dict, expected_fields: List[str]) -> bool:
    """Enhanced validation for IA responses"""
    if not response or not isinstance(response, dict):
        logging.error("Invalid response: not a dictionary")
        return False
    
    # Check required fields
    for field in expected_fields:
        if field not in response:
            logging.error(f"Missing required field in IA response: {field}")
            return False
    
    # Validate confidence score if present
    if 'confidence_score' in response:
        confidence = response['confidence_score']
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            logging.error(f"Invalid confidence score: {confidence}")
            return False
    
    # Validate symbol if present
    if 'symbol' in response:
        if not SecurityValidator.validate_symbol(response['symbol']):
            logging.error(f"Invalid symbol in response: {response['symbol']}")
            return False
    
    # Validate action if present
    if 'action' in response:
        valid_actions = {'buy', 'sell', 'long', 'short', 'hold', 'close'}
        if response['action'] not in valid_actions:
            logging.error(f"Invalid action in response: {response['action']}")
            return False
    
    return True

def execute_with_retry(func: Callable, max_retries: int = 3, delay: int = 2, **kwargs) -> Any:
    """Execute function with retry mechanism and enhanced error handling"""
    
    if TENACITY_AVAILABLE:
        # Use tenacity for more sophisticated retry logic
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=delay, max=10),
            retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError))
        )
        def _execute_with_tenacity():
            return func(**kwargs)
        
        return _execute_with_tenacity()
    
    else:
        # Fallback to simple retry logic
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func(**kwargs)
            except (requests.RequestException, ConnectionError, TimeoutError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)  # Exponential backoff
                    logging.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logging.error(f"All {max_retries} attempts failed")
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                raise
        
        raise last_exception

# Utility function for secure API calls
def call_ia_endpoint(url: str, payload: Dict, api_key: str = None, timeout: int = 30) -> requests.Response:
    """Make secure call to IA endpoint with validation"""
    
    # Validate inputs
    if not url or not isinstance(url, str):
        raise ValueError("Invalid URL provided")
    
    if not url.startswith(('http://', 'https://')):
        raise ValueError("URL must start with http:// or https://")
    
    # Sanitize payload
    if payload:
        sanitized_payload = SecurityValidator.sanitize_log_data(payload)
        logging.debug(f"Calling IA endpoint {url} with payload: {sanitized_payload}")
    
    # Prepare headers
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    # Make the request
    response = requests.post(
        url, 
        json=payload, 
        headers=headers,
        timeout=timeout
    )
    
    return response