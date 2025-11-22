import logging
import time

from django.conf import settings
from django.utils.deprecation import MiddlewareMixin


class RequestLoggingMiddleware(MiddlewareMixin):
    """
    Comprehensive middleware for logging all incoming requests and outgoing responses.
    Provides detailed context about each API call.
    """

    def __init__(self, get_response=None):
        super().__init__(get_response)
        self.get_response = get_response
        self.api_logger = logging.getLogger('users_api')
        self.performance_logger = logging.getLogger('performance')

    def __call__(self, request):
        # Skip logging for certain paths (health checks, static files, etc.)
        if self._should_skip_logging(request):
            return self.get_response(request)

        start_time = time.time()
        request_id = self._generate_request_id()

        # Store request info for context
        request.request_id = request_id
        request.start_time = start_time

        # Логируем входящий запрос
        self._log_request(request, request_id)

        # Process the request
        response = self.get_response(request)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Логируем исходящий ответ
        self._log_response(request, response, execution_time, request_id)

        # Log performance metrics
        self._log_performance(request, response, execution_time, request_id)

        return response

    @staticmethod
    def _should_skip_logging(request):
        """Determine if logging should be skipped for this request."""
        skip_paths = [
            '/health',
            '/favicon.ico',
            '/static/',
            '/media/',
            '/admin/',
        ]

        return any(request.path.startswith(path) for path in skip_paths)

    @staticmethod
    def _generate_request_id():
        """Generate a unique request ID for tracking."""
        import uuid
        return str(uuid.uuid4())[:8]

    @staticmethod
    def _get_client_ip(request):
        """Extract client IP address from request."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip

    @staticmethod
    def _get_user_info(request):
        """Extract user information from request."""
        if hasattr(request, 'user') and request.user.is_authenticated:
            return {
                'username': request.user.username,
                'user_id': request.user.id,
                'is_authenticated': True
            }
        return {
            'username': 'Anonymous',
            'user_id': None,
            'is_authenticated': False
        }

    def _log_request(self, request, request_id):
        """Log incoming request details."""
        user_info = self._get_user_info(request)
        client_ip = self._get_client_ip(request)

        log_data = {
            'request_id': request_id,
            'method': request.method,
            'path': request.path,
            'user': user_info['username'],
            'user_id': user_info['user_id'],
            'client_ip': client_ip,
            'user_agent': request.META.get('HTTP_USER_AGENT', 'Unknown'),
            'content_type': request.content_type,
        }

        # Log query parameters for GET requests
        if request.method == 'GET' and request.GET:
            log_data['query_params'] = dict(request.GET)

        self.api_logger.info(
            f"REQUEST {request_id} - {request.method} {request.path} - "
            f"User: {user_info['username']} - IP: {client_ip}",
            extra={'log_data': log_data}
        )

    def _log_response(self, request, response, execution_time, request_id):
        """Log outgoing response details."""
        user_info = self._get_user_info(request)

        log_data = {
            'request_id': request_id,
            'method': request.method,
            'path': request.path,
            'status_code': response.status_code,
            'user': user_info['username'],
            'user_id': user_info['user_id'],
            'execution_time': round(execution_time, 3),
            'content_type': getattr(response, 'content_type', 'Unknown'),
        }

        # Add additional info for error responses
        if response.status_code >= 400:
            log_data['error'] = True
            if hasattr(response, 'data') and isinstance(response.data, dict):
                log_data['error_details'] = response.data.get('error', 'Unknown error')

        self.api_logger.info(
            f"RESPONSE {request_id} - {request.method} {request.path} - "
            f"Status: {response.status_code} - Time: {execution_time:.3f}s - "
            f"User: {user_info['username']}",
            extra={'log_data': log_data}
        )

    def _log_performance(self, request, response, execution_time, request_id):
        """Log performance metrics for the request."""
        user_info = self._get_user_info(request)

        performance_data = {
            'request_id': request_id,
            'method': request.method,
            'path': request.path,
            'status_code': response.status_code,
            'user': user_info['username'],
            'execution_time': execution_time,
        }

        # Categorize performance
        if execution_time > 2.0:
            level = 'CRITICAL'
        elif execution_time > 1.0:
            level = 'WARNING'
        elif execution_time > 0.5:
            level = 'INFO'
        else:
            level = 'DEBUG'

        if level in ['WARNING', 'CRITICAL']:
            self.performance_logger.warning(
                f"PERFORMANCE {request_id} - {request.method} {request.path} - "
                f"Time: {execution_time:.3f}s - User: {user_info['username']}",
                extra={'performance_data': performance_data}
            )
        else:
            self.performance_logger.info(
                f"PERFORMANCE {request_id} - {request.method} {request.path} - "
                f"Time: {execution_time:.3f}s - User: {user_info['username']}",
                extra={'performance_data': performance_data}
            )

    def process_exception(self, request, exception):
        """Log exceptions that occur during request processing."""
        request_id = getattr(request, 'request_id', 'unknown')
        user_info = self._get_user_info(request)

        self.api_logger.error(
            f"EXCEPTION {request_id} - {request.method} {request.path} - "
            f"User: {user_info['username']} - Error: {str(exception)}",
            exc_info=True,
            extra={
                'exception_data': {
                    'request_id': request_id,
                    'method': request.method,
                    'path': request.path,
                    'user': user_info['username'],
                    'exception_type': type(exception).__name__,
                    'exception_message': str(exception),
                }
            }
        )


class DatabaseQueryLogger:
    """
    Middleware for logging database queries (optional, for debugging).
    """

    def __init__(self, get_response):
        self.get_response = get_response
        self.logger = logging.getLogger('performance')

    def __call__(self, request):
        from django.db import connection

        # Only log in debug mode or when explicitly enabled
        if not settings.DEBUG:
            return self.get_response(request)

        start_queries = len(connection.queries)
        start_time = time.time()

        response = self.get_response(request)

        end_time = time.time()
        total_queries = len(connection.queries) - start_queries
        execution_time = end_time - start_time

        if total_queries > 0:
            self.logger.debug(
                f"DB_QUERIES - {request.method} {request.path} - "
                f"Queries: {total_queries} - Time: {execution_time:.3f}s"
            )

        return response
