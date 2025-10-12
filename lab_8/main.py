import asyncio
import aiohttp
import ssl
from pathlib import Path
from datetime import datetime

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


async def scan_ports(domain: str, ports: list = None) -> dict:
    """
    Асинхронное сканирование портов домена
    """
    if ports is None:
        ports = COMMON_PORTS

    open_ports = []
    tasks = []

    # Создаем задачи для проверки каждого порта
    for port in ports:
        task = asyncio.create_task(check_port(domain, port))
        tasks.append(task)

    # Ждем завершения всех проверок портов
    port_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Собираем результаты
    for i, result in enumerate(port_results):
        if isinstance(result, dict) and result['is_open']:
            open_ports.append({
                'port': ports[i],
                'service': result.get('service', 'unknown'),
                'banner': result.get('banner', '')[:50]  # Ограничиваем длину баннера
            })

    return {
        'open_ports': open_ports,
        'total_scanned': len(ports),
        'open_count': len(open_ports)
    }


async def check_port(domain: str, port: int) -> dict:
    """
    Проверка конкретного порта
    """
    try:
        # Пытаемся подключиться к порту с таймаутом
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(domain, port),
            timeout=2.0
        )

        # Порт открыт - пробуем получить баннер
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

    except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
        return {
            'is_open': False,
            'port': port
        }
    except Exception as e:
        return {
            'is_open': False,
            'port': port,
            'error': str(e)
        }


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


async def get_ssl_info(domain: str) -> dict:
    """
    Получение информации о SSL-сертификате
    """
    try:
        context = ssl.create_default_context()
        reader, writer = await asyncio.open_connection(domain, 443, ssl=context)
        ssl_object = writer.get_extra_info('ssl_object')
        cert = ssl_object.getpeercert()

        writer.close()
        await writer.wait_closed()

        not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
        not_before = datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y %Z')
        days_until_expiry = (not_after - datetime.now()).days

        return {
            'has_ssl': True,
            'issuer': dict(x[0] for x in cert['issuer']),
            'subject': dict(x[0] for x in cert['subject']),
            'not_before': not_before,
            'not_after': not_after,
            'days_until_expiry': days_until_expiry,
            'is_valid': days_until_expiry > 0,
            'error': None
        }

    except Exception as e:
        return {
            'has_ssl': False,
            'error': str(e)
        }


async def check_site_availability(session: aiohttp.ClientSession, domain: str) -> dict:
    """
    Асинхронная проверка доступности сайта, SSL, security headers и портов
    """
    try:
        url = f"https://{domain}"
        async with session.get(url, timeout=10, ssl=False) as response:
            # Параллельно проверяем SSL, security headers и порты
            ssl_info_task = asyncio.create_task(get_ssl_info(domain))
            security_headers_task = asyncio.create_task(analyze_security_headers(response.headers))
            port_scan_task = asyncio.create_task(scan_ports(domain))

            ssl_info, security_headers_info, port_scan_info = await asyncio.gather(
                ssl_info_task, security_headers_task, port_scan_task
            )

            return {
                'domain': domain,
                'status': response.status,
                'available': True,
                'ssl_info': ssl_info,
                'security_headers': security_headers_info,
                'port_scan': port_scan_info,
                'error': None
            }
    except asyncio.TimeoutError:
        ssl_info = await get_ssl_info(domain)
        port_scan_info = await scan_ports(domain)
        return {
            'domain': domain,
            'status': 'timeout',
            'available': False,
            'ssl_info': ssl_info,
            'security_headers': None,
            'port_scan': port_scan_info,
            'error': 'Timeout (10s)'
        }
    except Exception as e:
        ssl_info = await get_ssl_info(domain)
        port_scan_info = await scan_ports(domain)
        return {
            'domain': domain,
            'status': 'error',
            'available': False,
            'ssl_info': ssl_info,
            'security_headers': None,
            'port_scan': port_scan_info,
            'error': str(e)
        }


async def main():
    """
    Основная асинхронная функция
    """
    print("🔄 Загрузка доменов из файла...")

    domains_file = Path("domains.txt")
    if not domains_file.exists():
        print("❌ Файл domains.txt не найден!")
        return

    domains = domains_file.read_text().strip().split('\n')
    domains = [d.strip() for d in domains if d.strip()]

    print(f"🔍 Найдено {len(domains)} доменов для проверки")
    print("⏳ Проверяем доступность, SSL, security headers и сканируем порты...\n")

    async with aiohttp.ClientSession() as session:
        tasks = [check_site_availability(session, domain) for domain in domains]
        results = await asyncio.gather(*tasks)

    # Вывод результатов
    successful = 0
    valid_ssl = 0
    good_security = 0
    total_open_ports = 0

    for result in results:
        ssl_status = "🔒" if result['ssl_info']['has_ssl'] else "🔓"

        if result['available']:
            successful += 1

            # Security headers информация
            security_info = result['security_headers']
            security_score = f" | Security: {security_info['score']}/{security_info['max_score']} ({security_info['rating']})"

            # SSL информация
            ssl_details = ""
            if result['ssl_info']['has_ssl']:
                valid_ssl += 1
                days = result['ssl_info']['days_until_expiry']
                ssl_details = f" | SSL: {days} дней"

            # Информация о портах
            port_info = result['port_scan']
            port_details = f" | Ports: {port_info['open_count']}/{port_info['total_scanned']} открыто"

            print(
                f"✅ {ssl_status} {result['domain']} - Доступен (Status: {result['status']}{ssl_details}{security_score}{port_details})")

            # Вывод открытых портов
            if port_info['open_ports']:
                open_ports_str = ", ".join([f"{p['port']}({p['service']})" for p in port_info['open_ports'][:3]])
                if len(port_info['open_ports']) > 3:
                    open_ports_str += f" ... (+{len(port_info['open_ports']) - 3})"
                print(f"   🔓 Открытые порты: {open_ports_str}")

            total_open_ports += port_info['open_count']

            # Подсчет сайтов с хорошей безопасностью
            if security_info['percentage'] >= 60:
                good_security += 1

        else:
            port_info = result['port_scan']
            port_details = f" | Ports: {port_info['open_count']}/{port_info['total_scanned']} открыто"
            ssl_error = f" | SSL: {result['ssl_info'].get('error', 'N/A')}" if not result['ssl_info']['has_ssl'] else ""
            print(f"❌ {ssl_status} {result['domain']} - Недоступен: {result['error']}{ssl_error}{port_details}")

    print(f"\n📊 Итоги:")
    print(f"   • {successful}/{len(domains)} сайтов доступно")
    print(f"   • {valid_ssl}/{len(domains)} имеют валидные SSL-сертификаты")
    print(f"   • {good_security}/{successful} сайтов с хорошей безопасностью headers")
    print(f"   • Найдено {total_open_ports} открытых портов")


if __name__ == "__main__":
    asyncio.run(main())