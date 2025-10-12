import asyncio
import aiohttp
from pathlib import Path


async def check_site_availability(session: aiohttp.ClientSession, domain: str) -> dict:
    """
    Асинхронная проверка доступности сайта
    """
    try:
        url = f"https://{domain}"
        async with session.get(url, timeout=10, ssl=False) as response:
            return {
                'domain': domain,
                'status': response.status,
                'available': True,
                'error': None
            }
    except asyncio.TimeoutError:
        return {
            'domain': domain,
            'status': 'timeout',
            'available': False,
            'error': 'Timeout (10s)'
        }
    except Exception as e:
        return {
            'domain': domain,
            'status': 'error',
            'available': False,
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
    print("⏳ Проверяем доступность сайтов...\n")

    # Асинхронная проверка всех доменов
    async with aiohttp.ClientSession() as session:
        tasks = [check_site_availability(session, domain) for domain in domains]
        results = await asyncio.gather(*tasks)

    # Вывод результатов
    successful = 0
    for result in results:
        if result['available']:
            print(f"✅ {result['domain']} - Доступен (Status: {result['status']})")
            successful += 1
        else:
            print(f"❌ {result['domain']} - Недоступен: {result['error']}")

    print(f"\n📊 Итоги: {successful}/{len(domains)} сайтов доступно")


if __name__ == "__main__":
    asyncio.run(main())