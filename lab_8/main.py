import asyncio
import aiohttp
import ssl
from pathlib import Path
from datetime import datetime


async def get_ssl_info(domain: str) -> dict:
    """
    Получение информации о SSL-сертификате
    """
    try:
        # Создаем SSL контекст
        context = ssl.create_default_context()

        # Подключаемся к домену и получаем сертификат
        reader, writer = await asyncio.open_connection(
            domain, 443, ssl=context
        )

        # Получаем SSL сертификат
        ssl_object = writer.get_extra_info('ssl_object')
        cert = ssl_object.getpeercert()

        writer.close()
        await writer.wait_closed()

        # Парсим даты валидности
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
    Асинхронная проверка доступности сайта и SSL
    """
    try:
        url = f"https://{domain}"
        async with session.get(url, timeout=10, ssl=False) as response:
            # Параллельно проверяем SSL
            ssl_info = await get_ssl_info(domain)

            return {
                'domain': domain,
                'status': response.status,
                'available': True,
                'ssl_info': ssl_info,
                'error': None
            }
    except asyncio.TimeoutError:
        ssl_info = await get_ssl_info(domain)  # Все равно пробуем получить SSL info
        return {
            'domain': domain,
            'status': 'timeout',
            'available': False,
            'ssl_info': ssl_info,
            'error': 'Timeout (10s)'
        }
    except Exception as e:
        ssl_info = await get_ssl_info(domain)  # Все равно пробуем получить SSL info
        return {
            'domain': domain,
            'status': 'error',
            'available': False,
            'ssl_info': ssl_info,
            'error': str(e)
        }


async def main():
    """
    Основная асинхронная функция
    """
    print("🔄 Загрузка доменов из файла...")

    # Чтение доменов из файла
    domains_file = Path("domains.txt")
    if not domains_file.exists():
        print("❌ Файл domains.txt не найден!")
        return

    domains = domains_file.read_text().strip().split('\n')
    domains = [d.strip() for d in domains if d.strip()]

    print(f"🔍 Найдено {len(domains)} доменов для проверки")
    print("⏳ Проверяем доступность сайтов и SSL-сертификаты...\n")

    # Асинхронная проверка всех доменов
    async with aiohttp.ClientSession() as session:
        tasks = [check_site_availability(session, domain) for domain in domains]
        results = await asyncio.gather(*tasks)

    # Вывод результатов
    successful = 0
    valid_ssl = 0

    for result in results:
        ssl_status = "🔒" if result['ssl_info']['has_ssl'] else "🔓"
        ssl_details = ""

        if result['ssl_info']['has_ssl']:
            valid_ssl += 1
            days = result['ssl_info']['days_until_expiry']
            ssl_details = f" | SSL: {days} дней"

        if result['available']:
            print(f"✅ {ssl_status} {result['domain']} - Доступен (Status: {result['status']}{ssl_details})")
            successful += 1
        else:
            ssl_error = f" | SSL: {result['ssl_info'].get('error', 'N/A')}" if not result['ssl_info']['has_ssl'] else ""
            print(f"❌ {ssl_status} {result['domain']} - Недоступен: {result['error']}{ssl_error}")

    print(f"\n📊 Итоги:")
    print(f"   • {successful}/{len(domains)} сайтов доступно")
    print(f"   • {valid_ssl}/{len(domains)} имеют валидные SSL-сертификаты")


if __name__ == "__main__":
    asyncio.run(main())