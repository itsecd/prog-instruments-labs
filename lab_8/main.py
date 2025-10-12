import asyncio
import aiohttp
import ssl
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Any
from functools import wraps
import sys
import time

# Настройка логирования только для ошибок
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Цвета для консоли
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# Конфигурация приложения
class Config:
    HTTP_TIMEOUT = 10
    PORT_SCAN_TIMEOUT = 2
    SSL_TIMEOUT = 5
    MAX_RETRIES = 2
    RETRY_DELAY = 1
    MAX_CONCURRENT_SCANS = 10


# Конфигурация проверяемых security headers
SECURITY_HEADERS = {
    'Content-Security-Policy': {'weight': 3, 'description': 'Защита от XSS'},
    'Strict-Transport-Security': {'weight': 3, 'description': 'Принудительное HTTPS'},
    'X-Content-Type-Options': {'weight': 2, 'description': 'Защита от MIME-sniffing'},
    'X-Frame-Options': {'weight': 2, 'description': 'Защита от clickjacking'},
    'X-XSS-Protection': {'weight': 1, 'description': 'Защита от XSS (устаревающая)'},
    'Referrer-Policy': {'weight': 1, 'description': 'Контроль реферера'},
    'Permissions-Policy': {'weight': 2, 'description': 'Контроль функций браузера'},
}

# Основные порты для сканирования
COMMON_PORTS = [
    21, 22, 23, 25, 53, 80, 110, 143, 443, 587, 993, 995,
    1433, 3306, 3389, 5432, 6379, 27017
]


class SecurityScannerError(Exception):
    """Базовое исключение для сканера безопасности"""
    pass


class TimeoutError(SecurityScannerError):
    """Таймаут операции"""
    pass


class ProgressBar:
    """Простой прогресс-бар для отображения хода выполнения"""

    def __init__(self, total: int, description: str = "Прогресс"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.width = 40

    def update(self, n: int = 1):
        """Обновить прогресс"""
        self.current += n
        self.display()

    def display(self):
        """Отобразить текущий прогресс"""
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)

        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            time_info = f"{elapsed:.1f}s / ~{eta:.1f}s"
        else:
            time_info = f"{elapsed:.1f}s"

        sys.stdout.write(
            f"\r{Colors.CYAN}{self.description}:{Colors.END} [{bar}] {self.current}/{self.total} ({percent:.1%}) {time_info}")
        sys.stdout.flush()

    def finish(self):
        """Завершить отображение прогресс-бара"""
        self.current = self.total
        self.display()
        print()  # Новая строка после завершения


def print_colored(text: str, color: str = Colors.WHITE, end: str = "\n"):
    """Печать цветного текста"""
    print(f"{color}{text}{Colors.END}", end=end)


def print_header(text: str):
    """Печать заголовка"""
    print_colored(f"\n{text}", Colors.BOLD + Colors.CYAN)
    print_colored("=" * len(text), Colors.CYAN)


def print_success(text: str):
    """Печать успешного сообщения"""
    print_colored(text, Colors.GREEN)


def print_warning(text: str):
    """Печать предупреждения"""
    print_colored(text, Colors.YELLOW)


def print_error(text: str):
    """Печать ошибки"""
    print_colored(text, Colors.RED)


def print_info(text: str):
    """Печать информационного сообщения"""
    print_colored(text, Colors.BLUE)


def get_service_name(port: int) -> str:
    """
    Получение имени сервиса по номеру порта
    """
    service_names = {
        21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
        80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS", 587: "SMTP SSL",
        993: "IMAP SSL", 995: "POP3 SSL", 1433: "MSSQL", 3306: "MySQL",
        3389: "RDP", 5432: "PostgreSQL", 6379: "Redis", 27017: "MongoDB"
    }
    return service_names.get(port, "unknown")


def get_security_rating(percentage: float) -> str:
    """
    Определение рейтинга безопасности по проценту
    """
    if percentage >= 80:
        return f"{Colors.GREEN}ОТЛИЧНО{Colors.END}"
    elif percentage >= 60:
        return f"{Colors.YELLOW}ХОРОШО{Colors.END}"
    elif percentage >= 40:
        return f"{Colors.YELLOW}УДОВЛЕТВОРИТЕЛЬНО{Colors.END}"
    else:
        return f"{Colors.RED}ПЛОХО{Colors.END}"


def get_ssl_status_color(days_until_expiry: int) -> str:
    """
    Получение цвета для статуса SSL сертификата
    """
    if days_until_expiry > 30:
        return Colors.GREEN
    elif days_until_expiry > 7:
        return Colors.YELLOW
    else:
        return Colors.RED


async def analyze_security_headers(response_headers) -> dict:
    """
    Анализ security headers из ответа сервера
    """
    security_score = 0
    max_score = sum(header['weight'] for header in SECURITY_HEADERS.values())
    found_headers = {}
    missing_headers = []

    for header, config in SECURITY_HEADERS.items():
        if header in response_headers:
            security_score += config['weight']
            found_headers[header] = {
                'value': response_headers[header],
                'description': config['description']
            }
        else:
            missing_headers.append(header)

    security_percentage = (security_score / max_score) * 100 if max_score > 0 else 0

    return {
        'score': security_score,
        'max_score': max_score,
        'percentage': security_percentage,
        'found_headers': found_headers,
        'missing_headers': missing_headers,
        'rating': get_security_rating(security_percentage)
    }


def async_retry(max_retries: int = Config.MAX_RETRIES, delay: float = Config.RETRY_DELAY):
    """
    Декоратор для повторных попыток выполнения асинхронных функций
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(delay)
                    else:
                        pass
            raise last_exception

        return wrapper

    return decorator


@async_retry()
async def scan_ports(domain: str, ports: List[int] = None) -> Dict[str, Any]:
    """
    Асинхронное сканирование портов домена с повторными попытками
    """
    if ports is None:
        ports = COMMON_PORTS

    open_ports = []

    async def check_port_with_timeout(port: int):
        return await check_port(domain, port)

    tasks = [check_port_with_timeout(port) for port in ports]
    port_results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(port_results):
        if isinstance(result, Exception):
            continue

        if isinstance(result, dict) and result['is_open']:
            open_ports.append({
                'port': ports[i],
                'service': result.get('service', 'unknown'),
                'banner': result.get('banner', '')[:50]
            })

    return {
        'open_ports': open_ports,
        'total_scanned': len(ports),
        'open_count': len(open_ports),
        'errors': len([r for r in port_results if isinstance(r, Exception)])
    }


@async_retry()
async def check_port(domain: str, port: int) -> Dict[str, Any]:
    """
    Проверка конкретного порта с таймаутом
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(domain, port),
            timeout=Config.PORT_SCAN_TIMEOUT
        )

        banner = ""
        try:
            writer.write(b"\r\n")
            await asyncio.wait_for(writer.drain(), timeout=1.0)
            banner_data = await asyncio.wait_for(reader.read(100), timeout=1.0)
            banner = banner_data.decode('utf-8', errors='ignore').strip()
        except:
            pass

        writer.close()
        await writer.wait_closed()

        return {
            'is_open': True,
            'port': port,
            'service': get_service_name(port),
            'banner': banner
        }

    except asyncio.TimeoutError:
        raise TimeoutError(f"Таймаут подключения к {domain}:{port}")
    except ConnectionRefusedError:
        return {'is_open': False, 'port': port}
    except OSError as e:
        raise SecurityScannerError(f"Ошибка OS при подключении к {domain}:{port}: {e}")
    except Exception as e:
        raise SecurityScannerError(f"Неожиданная ошибка при проверке порта {domain}:{port}: {e}")


@async_retry()
async def get_ssl_info(domain: str) -> Dict[str, Any]:
    """
    Получение информации о SSL-сертификате с обработкой ошибок
    """
    try:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(domain, 443, ssl=context),
            timeout=Config.SSL_TIMEOUT
        )

        ssl_object = writer.get_extra_info('ssl_object')
        if not ssl_object:
            raise SecurityScannerError("Не удалось получить SSL объект")

        cert = ssl_object.getpeercert()
        if not cert:
            raise SecurityScannerError("Не удалось получить SSL сертификат")

        writer.close()
        await writer.wait_closed()

        not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
        not_before = datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y %Z')
        days_until_expiry = (not_after - datetime.now()).days

        issuer_info = dict(x[0] for x in cert['issuer'])
        issuer_name = issuer_info.get('organizationName', issuer_info.get('commonName', 'Unknown'))

        return {
            'has_ssl': True,
            'issuer': issuer_name,
            'subject': dict(x[0] for x in cert['subject']),
            'not_before': not_before.strftime('%Y-%m-%d'),
            'not_after': not_after.strftime('%Y-%m-%d'),
            'days_until_expiry': days_until_expiry,
            'is_valid': days_until_expiry > 0,
            'error': None
        }

    except asyncio.TimeoutError:
        raise TimeoutError(f"Таймаут SSL проверки для {domain}")
    except Exception as e:
        return {
            'has_ssl': False,
            'error': f"{type(e).__name__}: {str(e)}"
        }


@async_retry()
async def check_site_availability(session: aiohttp.ClientSession, domain: str) -> Dict[str, Any]:
    """
    Асинхронная проверка доступности сайта с улучшенной обработкой ошибок
    """
    try:
        timeout = aiohttp.ClientTimeout(total=Config.HTTP_TIMEOUT)
        url = f"https://{domain}"

        async with session.get(url, timeout=timeout, ssl=False) as response:
            ssl_info_task = asyncio.create_task(get_ssl_info(domain))
            security_headers_task = asyncio.create_task(analyze_security_headers(response.headers))
            port_scan_task = asyncio.create_task(scan_ports(domain))

            ssl_info, security_headers_info, port_scan_info = await asyncio.gather(
                ssl_info_task, security_headers_task, port_scan_task,
                return_exceptions=True
            )

            if isinstance(ssl_info, Exception):
                ssl_info = {'has_ssl': False, 'error': str(ssl_info)}

            if isinstance(security_headers_info, Exception):
                security_headers_info = None

            if isinstance(port_scan_info, Exception):
                port_scan_info = {'open_ports': [], 'total_scanned': 0, 'open_count': 0, 'errors': 1}

            return {
                'domain': domain,
                'status': response.status,
                'available': True,
                'ssl_info': ssl_info if not isinstance(ssl_info, Exception) else {'has_ssl': False,
                                                                                  'error': str(ssl_info)},
                'security_headers': security_headers_info if not isinstance(security_headers_info, Exception) else None,
                'port_scan': port_scan_info if not isinstance(port_scan_info, Exception) else {'open_ports': [],
                                                                                               'total_scanned': 0,
                                                                                               'open_count': 0,
                                                                                               'errors': 1},
                'error': None
            }

    except asyncio.TimeoutError:
        raise TimeoutError(f"Таймаут HTTP запроса для {domain}")
    except aiohttp.ClientError as e:
        raise SecurityScannerError(f"Ошибка HTTP клиента для {domain}: {e}")
    except Exception as e:
        raise SecurityScannerError(f"Неожиданная ошибка при проверке {domain}: {e}")


async def safe_check_domain(session: aiohttp.ClientSession, domain: str) -> Dict[str, Any]:
    """
    Безопасная проверка домена с обработкой всех исключений
    """
    try:
        return await check_site_availability(session, domain)
    except TimeoutError as e:
        ssl_info = await get_ssl_info(domain)
        port_scan = await scan_ports(domain)
        return {
            'domain': domain,
            'status': 'timeout',
            'available': False,
            'ssl_info': ssl_info,
            'security_headers': None,
            'port_scan': port_scan,
            'error': str(e)
        }
    except SecurityScannerError as e:
        return {
            'domain': domain,
            'status': 'error',
            'available': False,
            'ssl_info': {'has_ssl': False, 'error': 'Scanning failed'},
            'security_headers': None,
            'port_scan': {'open_ports': [], 'total_scanned': 0, 'open_count': 0, 'errors': 1},
            'error': str(e)
        }
    except Exception as e:
        return {
            'domain': domain,
            'status': 'critical_error',
            'available': False,
            'error': f"Critical: {type(e).__name__}: {str(e)}"
        }


async def main():
    """
    Основная асинхронная функция с цветным выводом и прогресс-баром
    """
    print_header("🔄 АСИНХРОННЫЙ СКАНЕР БЕЗОПАСНОСТИ")

    domains_file = Path("domains.txt")
    if not domains_file.exists():
        print_error("❌ Файл domains.txt не найден!")
        return

    domains = domains_file.read_text().strip().split('\n')
    domains = [d.strip() for d in domains if d.strip()]

    print_info(f"🔍 Найдено {len(domains)} доменов для проверки")
    print_info("⏳ Запускаем сканирование...")

    # Создаем прогресс-бар
    progress = ProgressBar(len(domains), "Сканирование доменов")

    results = []
    semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_SCANS)

    async def bounded_check(session, domain):
        async with semaphore:
            result = await safe_check_domain(session, domain)
            progress.update()
            return result

    async with aiohttp.ClientSession() as session:
        tasks = [bounded_check(session, domain) for domain in domains]

        # Запускаем все задачи и обновляем прогресс
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)

    progress.finish()

    # Вывод результатов
    print_header("📊 РЕЗУЛЬТАТЫ СКАНИРОВАНИЯ")

    successful = 0
    valid_ssl = 0
    good_security = 0
    total_open_ports = 0
    total_errors = 0

    for result in results:
        ssl_info = result.get('ssl_info', {})

        if result.get('available', False):
            successful += 1

            # Основная информация о сайте
            status_emoji = "✅" if result.get('status') == 200 else "⚠️"
            print_success(f"{status_emoji} {result['domain']} - Доступен (Status: {result.get('status', 'N/A')})")

            # SSL информация
            if ssl_info.get('has_ssl', False):
                valid_ssl += 1
                days = ssl_info.get('days_until_expiry', 0)
                issuer = ssl_info.get('issuer', 'Unknown')
                not_after = ssl_info.get('not_after', 'Unknown')
                ssl_color = get_ssl_status_color(days)

                print_colored(f"   📜 SSL: {issuer}", ssl_color)
                print_colored(f"   📅 Действует до: {not_after} ({days} дней)", ssl_color)
            else:
                ssl_error = ssl_info.get('error', 'No SSL')
                print_error(f"   ❌ SSL: {ssl_error}")

            # Security headers
            security_info = result.get('security_headers', {})
            if security_info:
                rating = security_info.get('rating', 'N/A')
                score = security_info.get('score', 0)
                max_score = security_info.get('max_score', 0)
                print_colored(f"   🛡️  Security Headers: {score}/{max_score} баллов - {rating}", Colors.MAGENTA)

            # Порты
            port_info = result.get('port_scan', {})
            open_ports = port_info.get('open_ports', [])
            if open_ports:
                total_open_ports += len(open_ports)
                ports_str = ", ".join([f"{p['port']}({p['service']})" for p in open_ports[:5]])
                if len(open_ports) > 5:
                    ports_str += f" ... (+{len(open_ports) - 5})"
                print_warning(f"   🔓 Открытые порты: {ports_str}")
            else:
                print_success("   🔒 Открытые порты: нет")

            if security_info and security_info.get('percentage', 0) >= 60:
                good_security += 1

        else:
            total_errors += 1
            error_msg = result.get('error', 'Unknown error')
            print_error(f"❌ {result['domain']} - Недоступен: {error_msg}")

            # SSL информация для недоступных сайтов
            if ssl_info.get('has_ssl', False):
                days = ssl_info.get('days_until_expiry', 0)
                issuer = ssl_info.get('issuer', 'Unknown')
                ssl_color = get_ssl_status_color(days)
                print_colored(f"   📜 SSL: {issuer} ({days} дней осталось)", ssl_color)

        print()  # Пустая строка между сайтами

    # Сводка
    print_header("📈 СВОДКА СКАНИРОВАНИЯ")

    print_colored(f"   📊 Общее количество доменов: {len(domains)}", Colors.BOLD)
    print_success(f"   ✅ Доступных сайтов: {successful}/{len(domains)}")
    print_colored(f"   🔐 SSL сертификатов: {valid_ssl}/{len(domains)}",
                  Colors.GREEN if valid_ssl == len(domains) else Colors.YELLOW)
    print_colored(f"   🛡️  С хорошей безопасностью: {good_security}/{successful}",
                  Colors.GREEN if good_security == successful else Colors.YELLOW)
    print_warning(f"   🔓 Открытых портов: {total_open_ports}")
    print_error(f"   ❌ Ошибок сканирования: {total_errors}")

    # Общая оценка
    success_rate = (successful / len(domains)) * 100
    security_rate = (good_security / successful * 100) if successful > 0 else 0

    print_header("🏆 ОБЩАЯ ОЦЕНКА")

    if success_rate >= 80 and security_rate >= 80:
        print_success("   🎉 Отличные показатели безопасности!")
    elif success_rate >= 60 and security_rate >= 60:
        print_colored("   👍 Хорошие показатели безопасности", Colors.YELLOW)
    else:
        print_error("   ⚠️  Есть проблемы с безопасностью")

    print_colored(f"   📈 Успешность сканирования: {success_rate:.1f}%", Colors.CYAN)
    print_colored(f"   🛡️  Уровень безопасности: {security_rate:.1f}%", Colors.CYAN)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_error("\n⚠️  Сканирование прервано пользователем")
    except Exception as e:
        print_error(f"💥 Критическая ошибка приложения: {e}")