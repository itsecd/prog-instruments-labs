import argparse
import csv
import os
import random
import requests
from bs4 import BeautifulSoup
from iter import SoundtrackIterator
import logging

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="downloads")
    parser.add_argument("-c", "--csv", type=str, default="downloads/ann.csv")
    parser.add_argument("-n", "--number", type=int, default=50)
    return parser.parse_args()


def collect_tracks(genre: str) -> list[dict[str, str]]:
    """
    Извлекает информацию о треках с сайта mixkit.co
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }

    if genre == "R&B":
        url_genre = "contemporary-r-and-b"
    else:
        url_genre = genre
    
    url = f"https://mixkit.co/free-stock-music/{url_genre}/"
    logger.debug("Fetching %s", url)
    try:
        response = requests.get(url, headers=headers, timeout=40)
        response.raise_for_status()
    except requests.RequestException:
        logger.exception("Failed to fetch %s", url)
        raise
    logger.info("Fetched %s: status=%d bytes=%d", url, response.status_code, len(response.content))
    soup = BeautifulSoup(response.text, "html.parser")
    cards = soup.find_all("div", class_="item-grid-card item-grid-card--show-meta")

    tracks = []
    for card in cards:
        name_tag = card.find("h2")
        author_tag = card.find("p")
        audio_tag = card.find("div", attrs={"data-audio-player-preview-url-value": True})
        
        if not (name_tag and author_tag and audio_tag):
            continue
            
        tracks.append({
            "title": name_tag.get_text(strip=True),
            "artist": author_tag.get_text(strip=True).replace("by ", ""),
            "link": audio_tag["data-audio-player-preview-url-value"],
        })
    return tracks


def download_audio_track(gen_fol: str, url: str) -> str:
    """Скачивает аудиофайл в папку жанра"""
    os.makedirs(gen_fol, exist_ok=True)
    filename = os.path.basename(url)
    filepath = os.path.join(gen_fol, filename)

    if not os.path.exists(filepath):
        logger.debug("Downloading %s -> %s", url, filepath)
        try:
            with requests.get(url, stream=True, timeout=40) as response:
                response.raise_for_status()
                with open(filepath, "wb") as f:
                    for data_part in response.iter_content(8192):
                        f.write(data_part)
        except requests.RequestException:
            logger.exception("Download failed: %s", url)
            raise
        logger.info("Downloaded %s", filename)
    return filename


def handle_genre(genre: str, base_dir: str, writer: csv.writer, tracks_per_genre: int) -> int:
    """Обрабатывает один жанр: парсит, скачивает треки и записывает в CSV"""
    logger.info("Start genre: %s", genre)
    try:
        all_tracks = collect_tracks(genre)
    except requests.RequestException as e:
        logger.error("Request failed for genre %s: %s", genre, e)
        return 0

    if not all_tracks:
        logger.warning("No tracks for genre: %s", genre)
        return 0

    count = min(tracks_per_genre, len(all_tracks))
    selected = random.sample(all_tracks, count)

    logger.info("Will download %d tracks for genre %s", count, genre)
    genre_dir = os.path.join(base_dir, genre)
    downloaded_count = 0
    failed_count = 0 
    for track in selected:
        try:
            fname = download_audio_track(genre_dir, track["link"])
            rel_path = f"{genre}/{fname}"
            abs_path = os.path.abspath(os.path.join(genre_dir, fname))
            writer.writerow([rel_path, abs_path, genre, track["title"], track["artist"]])
            downloaded_count += 1
        except requests.RequestException:
            failed_count += 1        

    logger.info("Genre %s done: downloaded=%d failed=%d",
                genre, downloaded_count, failed_count)
    return downloaded_count


def run_downloads(out_dir: str, csv_file: str, genres: list[str]) -> dict[str, int]:
    """Организует загрузку треков по всем жанрам и формирует CSV-аннотацию"""

    args = get_args()
    total_tracks = max(50, min(1000, args.number))
    
    tracks_per_genre = total_tracks // len(genres)
    logger.info("Total tracks: %d, per genre: %d", total_tracks, tracks_per_genre)
    results = {"genres_processed": 0, "tracks_total": 0}
    
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Относительный путь", "Абсолютный путь", "Жанр", "Название", "Исполнитель"])

        for genre in genres:
            downloaded = handle_genre(genre, out_dir, writer, tracks_per_genre)
            if downloaded:
                results["genres_processed"] += 1
                results["tracks_total"] += downloaded
    logger.info(
        "Run finished: genres=%d tracks=%d",
        results["genres_processed"], results["tracks_total"],
    )
    return results


def print_summary(csv_file: str, stats: dict[str, int]) -> None:
    """Вывод статистики и предпросмотра содержимого CSV."""
    logger.info("Finished: tracks=%d", stats["tracks_total"])
    logger.debug("CSV preview from %s", csv_file)
    for row in SoundtrackIterator(csv_file):
        logger.debug("Row: %s", row)