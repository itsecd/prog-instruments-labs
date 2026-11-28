"""
Асинхронный нагрузочный тестер для сайтов.

Использует aiohttp для выполнения HTTP/HTTPS запросов с заданной частотой (RPS)
и собирает статистику по времени ответа, статус-кодам и ошибкам.
"""

import asyncio
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import aiohttp
from aiohttp import ClientTimeout, ClientConnectionError, ClientConnectorError


@dataclass
class TestConfig:
    """Конфигурация нагрузочного теста."""
    url: str
    rps: int
    test_time: int
    wait_timeout: int
    max_connections: int = 1000


@dataclass
class RequestResult:
    """Результат одного запроса."""
    success: bool
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class TestResults:
    """Агрегированные результаты теста."""
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    timeout_errors: int = 0
    connection_errors: int = 0
    http_errors: int = 0
    response_times: List[float] = field(default_factory=list)
    status_codes: Dict[int, int] = field(default_factory=dict)


class LoadTester:
    """Асинхронный нагрузочный тестер."""

    def __init__(self, config: TestConfig):
        self.config = config
        self.results = TestResults()
        self._semaphore = asyncio.Semaphore(config.max_connections)

    async def make_request(self, session: aiohttp.ClientSession) -> RequestResult:
        """Выполняет один HTTP-запрос и возвращает результат."""
        start_time = time.perf_counter()
        
        try:
            async with self._semaphore:
                async with session.get(self.config.url) as response:
                    await response.read()
                    response_time = (time.perf_counter() - start_time) * 1000
                    
                    return RequestResult(
                        success=True,
                        status_code=response.status,
                        response_time_ms=response_time
                    )
        
        except asyncio.TimeoutError:
            response_time = (time.perf_counter() - start_time) * 1000
            return RequestResult(
                success=False,
                response_time_ms=response_time,
                error_type="timeout",
                error_message="Request timed out"
            )
        
        except (ClientConnectionError, ClientConnectorError) as e:
            response_time = (time.perf_counter() - start_time) * 1000
            return RequestResult(
                success=False,
                response_time_ms=response_time,
                error_type="connection",
                error_message=str(e)
            )
        
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            return RequestResult(
                success=False,
                response_time_ms=response_time,
                error_type="other",
                error_message=str(e)
            )

    async def run_test(self):
        """Запускает нагрузочный тест."""
        print(f"Target URL: {self.config.url}")
        print(f"Test duration: {self.config.test_time}s, RPS: {self.config.rps}")
        print(f"Total requests: {self.config.rps * self.config.test_time}")
        print(f"Timeout for responses: {self.config.wait_timeout}s")
        print()

        timeout = ClientTimeout(total=self.config.wait_timeout)
        connector = aiohttp.TCPConnector(limit=self.config.max_connections)
        
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": "LoadTester/1.0"}
        ) as session:
            
            delay = 1.0 / self.config.rps
            total_requests = self.config.rps * self.config.test_time
            
            tasks = []
            start_time = time.perf_counter()
            
            for i in range(total_requests):
                expected_time = i * delay
                elapsed = time.perf_counter() - start_time
                wait_time = expected_time - elapsed
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                task = asyncio.create_task(self.make_request(session))
                tasks.append(task)
            
            done, pending = await asyncio.wait(
                tasks,
                timeout=self.config.wait_timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            for task in pending:
                task.cancel()
            
            print("Test completed\n")
        
        self._aggregate_results(done)

    def _aggregate_results(self, completed_tasks):
        """Агрегирует результаты выполненных задач."""
        for task in completed_tasks:
            try:
                result = task.result()
                self.results.total_requests += 1
                
                if result.success:
                    self.results.successful += 1
                    self.results.response_times.append(result.response_time_ms)
                    
                    if result.status_code:
                        self.results.status_codes[result.status_code] = \
                            self.results.status_codes.get(result.status_code, 0) + 1
                        
                        if result.status_code >= 500:
                            self.results.http_errors += 1
                else:
                    self.results.failed += 1
                    self.results.response_times.append(result.response_time_ms)
                    
                    if result.error_type == "timeout":
                        self.results.timeout_errors += 1
                    elif result.error_type == "connection":
                        self.results.connection_errors += 1
            
            except asyncio.CancelledError:
                self.results.total_requests += 1
                self.results.failed += 1
                self.results.timeout_errors += 1

    def print_report(self):
        """Выводит отчет о тесте."""
        print("=== Request Summary ===")
        print(f"Total requests sent:    {self.results.total_requests}")
        print(f"Completed successfully: {self.results.successful}")
        print(f"Failed requests:        {self.results.failed}")
        print(f"  - Timeout errors:     {self.results.timeout_errors}")
        print(f"  - Connection errors:  {self.results.connection_errors}")
        print(f"  - HTTP errors (5xx):  {self.results.http_errors}")
        print()
        
        if self.results.response_times:
            sorted_times = sorted(self.results.response_times)
            n = len(sorted_times)
            
            print("=== Response Time (ms) ===")
            print(f"Min:     {min(sorted_times):.0f}")
            print(f"Max:     {max(sorted_times):.0f}")
            print(f"Mean:    {statistics.mean(sorted_times):.1f}")
            print(f"Median:  {statistics.median(sorted_times):.0f}")
            
            p75 = sorted_times[int(n * 0.75)]
            p90 = sorted_times[int(n * 0.90)]
            p95 = sorted_times[int(n * 0.95)]
            p99 = sorted_times[int(n * 0.99)]
            
            print(f"p75:     {p75:.0f}")
            print(f"p90:     {p90:.0f}")
            print(f"p95:     {p95:.0f}")
            print(f"p99:     {p99:.0f}")
            print()
        
        print("=== Status Code Distribution ===")
        if self.results.status_codes:
            for code, count in sorted(self.results.status_codes.items()):
                description = self._get_status_description(code)
                print(f"{code} {description}: {count}")
        
        if self.results.timeout_errors > 0:
            print(f"Timeouts: {self.results.timeout_errors}")
        
        print()


    @staticmethod
    def _get_status_description(code: int) -> str:
        """Возвращает описание статус-кода."""
        descriptions = {
            200: "OK",
            201: "Created",
            301: "Moved Permanently",
            302: "Found",
            304: "Not Modified",
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            500: "Internal Server Error",
            502: "Bad Gateway",
            503: "Service Unavailable",
            504: "Gateway Timeout",
        }
        return descriptions.get(code, "")


def parse_args():
    """Парсит аргументы командной строки."""
    parser = ArgumentParser(
        description="Асинхронный нагрузочный тестер для сайтов",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python load_tester.py https://example.com --rps 10 --time 5
  python load_tester.py https://bitrix24.ru/prices/ --rps 100 --time 10 --timeout 5
        """
    )
    
    parser.add_argument(
        "url",
        help="URL сайта для тестирования"
    )
    parser.add_argument(
        "--rps",
        type=int,
        default=10,
        help="Количество запросов в секунду (по умолчанию: 10)"
    )
    parser.add_argument(
        "--time",
        type=int,
        default=10,
        help="Длительность теста в секундах (по умолчанию: 10)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Таймаут ожидания ответов после окончания теста (по умолчанию: 5)"
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=1000,
        help="Максимальное количество одновременных соединений (по умолчанию: 1000)"
    )
    
    return parser.parse_args()


async def main():
    """Главная функция."""
    args = parse_args()
    
    config = TestConfig(
        url=args.url,
        rps=args.rps,
        test_time=args.time,
        wait_timeout=args.timeout,
        max_connections=args.max_connections
    )
    
    tester = LoadTester(config)
    
    try:
        await tester.run_test()
        tester.print_report()
    except KeyboardInterrupt:
        print("\nТест прерван пользователем")
    except Exception as e:
        print(f"\nОшибка: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())