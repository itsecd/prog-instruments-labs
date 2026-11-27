import asyncio
import aiohttp
import csv
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CHECK_INTERVAL = 10

async def check_site(session, url, previous_status=None):
    try:
        async with session.head(url, timeout=aiohttp.ClientTimeout(total=5), allow_redirects=True) as response:
            status = response.status
            logger.debug(f"{url} -> {status}")
            return status
    except asyncio.TimeoutError:
        logger.warning(f"{url} -> ТАЙМАУТ")
        return None
    except aiohttp.ClientError as e:
        logger.warning(f"{url} -> ОШИБКА СОЕДИНЕНИЯ: {e}")
        return None

async def monitor_sites(sites, interval):
    prev_statuses = {}
    
    async with aiohttp.ClientSession() as session:
        while True:
            logger.info(f"--- Начало проверки {len(sites)} сайтов ---")
            
            tasks = [check_site(session, url, prev_statuses.get(url)) for url in sites]
            results = await asyncio.gather(*tasks)
            
            for url, new_status in zip(sites, results):
                old_status = prev_statuses.get(url)
                
                if old_status is not None and new_status != old_status:
                    if new_status is None:
                        logger.error(f"ALERT! {url} УПАЛ! (было: {old_status})")
                    elif old_status is None:
                        logger.info(f"ALERT! {url} ВОССТАНОВЛЕН! (теперь: {new_status})")
                    else:
                        logger.warning(f"ALERT! {url} изменил статус: {old_status} -> {new_status}")
                
                prev_statuses[url] = new_status
            
            logger.info(f"--- Проверка завершена. Ждём {interval} сек... ---")
            await asyncio.sleep(interval)

async def main():
    sites = []
    try:
        with open("sites.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                domain = row["DOMAIN_NAME"].strip()
                if domain:
                    if not domain.startswith("http"):
                        domain = "https://" + domain
                    sites.append(domain)
    except FileNotFoundError:
        logger.critical("Файл sites.csv не найден!")
        return
    
    if not sites:
        logger.critical("Список сайтов пуст!")
        return
    
    logger.info(f"Загружено сайтов: {len(sites)}")
    
    await monitor_sites(sites, CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Мониторинг остановлен пользователем.")
