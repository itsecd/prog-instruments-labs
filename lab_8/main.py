import asyncio
import aiohttp
import ssl
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Any
from functools import wraps

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    21,  # FTP
    22,  # SSH
    23,  # Telnet
    25,  # SMTP
    53,  # DNS
    80,  # HTTP
    110,  # POP3
    143,  # IMAP
    443,  # HTTPS
    587,  # SMTP SSL
    993,  # IMAP SSL
    995,  # POP3 SSL
    1433,  # MSSQL
    3306,  # MySQL
    3389,  # RDP
    5432,  # PostgreSQL
    6379,  # Redis
    27017,  # MongoDB
]


class SecurityScannerError(Exception):
    """Базовое исключение для сканера безопасности"""
    pass


class TimeoutError(SecurityScannerError):
    """Таймаут операции"""
    pass


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
        return "🟢 ОТЛИЧНО"
    elif percentage >= 60:
        return "🟡 ХОРОШО"
    elif percentage >= 40:
        return "🟠 УДОВЛЕТВОРИТЕЛЬНО"
    else:
        return "🔴 ПЛОХО"


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
                        logger.debug(f"Попытка {attempt + 1} не удалась, повтор через {delay}сек: {e}")
                        await asyncio.sleep(delay)
                    else:
                        logger.warning(f"Все {max_retries + 1} попыток не удались для {func.__name__}")
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
    semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_SCANS)

    async def check_port_with_semaphore(port: int):
        async with semaphore:
            return await check_port(domain, port)

    # Создаем задачи для проверки каждого порта
    tasks = [check_port_with_semaphore(port) for port in ports]

    # Ждем завершения всех проверок портов
    port_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Собираем результаты
    for i, result in enumerate(port_results):
        if isinstance(result, Exception):
            logger.error(f"Ошибка при сканировании порта {ports[i]} для {domain}: {result}")
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
        # Пытаемся подключиться к порту с таймаутом
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(domain, port),
            timeout=Config.PORT_SCAN_TIMEOUT
        )

        # Порт открыт - пробуем получить баннер
        banner = ""
        try:
            writer.write(b"\r\n")
            await asyncio.wait_for(writer.drain(), timeout=1.0)
            banner_data = await asyncio.wait_for(reader.read(100), timeout=1.0)
            banner = banner_data.decode('utf-8', errors='ignore').strip()
        except Exception as banner_error:
            logger.debug(f"Не удалось получить баннер для {domain}:{port}: {banner_error}")

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

        return {
            'has_ssl': True,
            'issuer': dict(x[0] for x in cert['issuer']),
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
            # Параллельно проверяем SSL, security headers и порты
            ssl_info_task = asyncio.create_task(get_ssl_info(domain))
            security_headers_task = asyncio.create_task(analyze_security_headers(response.headers))
            port_scan_task = asyncio.create_task(scan_ports(domain))

            ssl_info, security_headers_info, port_scan_info = await asyncio.gather(
                ssl_info_task, security_headers_task, port_scan_task,
                return_exceptions=True
            )

            # Обрабатываем возможные исключения в задачах
            if isinstance(ssl_info, Exception):
                logger.error(f"Ошибка SSL проверки для {domain}: {ssl_info}")
                ssl_info = {'has_ssl': False, 'error': str(ssl_info)}

            if isinstance(security_headers_info, Exception):
                logger.error(f"Ошибка анализа headers для {domain}: {security_headers_info}")
                security_headers_info = None

            if isinstance(port_scan_info, Exception):
                logger.error(f"Ошибка сканирования портов для {domain}: {port_scan_info}")
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
        logger.warning(f"Таймаут для {domain}: {e}")
        # Вызываем функции напрямую, а не через await в словаре
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
        logger.error(f"Ошибка сканирования {domain}: {e}")
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
        logger.error(f"Критическая ошибка для {domain}: {e}")
        return {
            'domain': domain,
            'status': 'critical_error',
            'available': False,
            'error': f"Critical: {type(e).__name__}: {str(e)}"
        }


async def main():
    """
    Основная асинхронная функция с улучшенной обработкой ошибок
    """
    print("🔄 Загрузка доменов из файла...")

    domains_file = Path("domains.txt")
    if not domains_file.exists():
        print("❌ Файл domains.txt не найден!")
        return

    domains = domains_file.read_text().strip().split('\n')
    domains = [d.strip() for d in domains if d.strip()]

    print(f"🔍 Найдено {len(domains)} доменов для проверки")
    print("⏳ Проверяем с улучшенной обработкой ошибок и ретраями...\n")

    # Семафор для ограничения одновременных запросов
    semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_SCANS)

    async def bounded_check(session, domain):
        async with semaphore:
            return await safe_check_domain(session, domain)

    async with aiohttp.ClientSession() as session:
        tasks = [bounded_check(session, domain) for domain in domains]
        results = await asyncio.gather(*tasks)

    # Вывод результатов
    successful = 0
    valid_ssl = 0
    good_security = 0
    total_open_ports = 0
    total_errors = 0

    for result in results:
        ssl_status = "🔒" if result.get('ssl_info', {}).get('has_ssl', False) else "🔓"

        if result.get('available', False):
            successful += 1

            # Security headers информация
            security_info = result.get('security_headers', {})
            security_score = ""
            if security_info:
                security_score = f" | Security: {security_info.get('score', 0)}/{security_info.get('max_score', 0)} ({security_info.get('rating', 'N/A')})"

            # SSL информация
            ssl_details = ""
            ssl_info = result.get('ssl_info', {})
            if ssl_info.get('has_ssl', False):
                valid_ssl += 1
                days = ssl_info.get('days_until_expiry', 0)
                ssl_details = f" | SSL: {days} дней"

            # Информация о портах
            port_info = result.get('port_scan', {})
            port_details = f" | Ports: {port_info.get('open_count', 0)}/{port_info.get('total_scanned', 0)} открыто"

            print(
                f"✅ {ssl_status} {result['domain']} - Доступен (Status: {result.get('status', 'N/A')}{ssl_details}{security_score}{port_details})")

            # Вывод открытых портов
            open_ports = port_info.get('open_ports', [])
            if open_ports:
                open_ports_str = ", ".join([f"{p['port']}({p['service']})" for p in open_ports[:3]])
                if len(open_ports) > 3:
                    open_ports_str += f" ... (+{len(open_ports) - 3})"
                print(f"   🔓 Открытые порты: {open_ports_str}")

            total_open_ports += port_info.get('open_count', 0)

            # Подсчет сайтов с хорошей безопасностью
            if security_info and security_info.get('percentage', 0) >= 60:
                good_security += 1

        else:
            port_info = result.get('port_scan', {})
            port_details = f" | Ports: {port_info.get('open_count', 0)}/{port_info.get('total_scanned', 0)} открыто"
            ssl_error = f" | SSL: {result.get('ssl_info', {}).get('error', 'N/A')}" if not result.get('ssl_info',
                                                                                                      {}).get('has_ssl',
                                                                                                              False) else ""
            print(
                f"❌ {ssl_status} {result['domain']} - Недоступен: {result.get('error', 'Unknown error')}{ssl_error}{port_details}")
            total_errors += 1

    print(f"\n📊 Итоги:")
    print(f"   • {successful}/{len(domains)} сайтов доступно")
    print(f"   • {valid_ssl}/{len(domains)} имеют валидные SSL-сертификаты")
    print(f"   • {good_security}/{successful} сайтов с хорошей безопасностью headers")
    print(f"   • Найдено {total_open_ports} открытых портов")
    print(f"   • Произошло {total_errors} ошибок при сканировании")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Сканирование прервано пользователем")
    except Exception as e:
        print(f"💥 Критическая ошибка приложения: {e}")
        logger.exception("Критическая ошибка:")