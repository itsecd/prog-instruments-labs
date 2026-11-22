"""Download images of pigs with BingImageCrawler and build a CSV annotation.

Lab work #5 "Logging": all diagnostic output is produced by the standard
`logging` library instead of print(). Configuration lives in config/logging.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import logging.config
import os
import time
from pathlib import Path
from typing import List

from icrawler.builtin import BingImageCrawler

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)

KEYWORD = "pig"
TARGET_COUNT = 50
BATCH_SIZE = 100
SLOW_BATCH_SECONDS = 20.0


def setup_logging() -> None:
    """Apply the JSON logging configuration (kept outside the code)."""
    (BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)
    config_path = Path(os.environ.get("LOG_CONFIG", BASE_DIR / "config" / "logging.json"))

    try:
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, ValueError) as exc:
        logging.basicConfig(level=logging.INFO)
        logger.error(
            "Cannot load logging configuration from %s (%s); falling back to console INFO logging",
            config_path, exc,
        )
        return

    for handler in config.get("handlers", {}).values():
        name = handler.get("filename")
        if name and not os.path.isabs(name):
            handler["filename"] = str(BASE_DIR / name)

    logging.config.dictConfig(config)
    logger.debug("Logging configured from %s", config_path)


def parsing() -> tuple[str, str, int]:
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description="Download pig images and annotate them")
    parser.add_argument("file_path", type=str, help="Directory to store downloaded images")
    parser.add_argument("annotation_path", type=str, help="Path to the annotation CSV file")
    parser.add_argument("--time_limit", type=int, default=30, help="Time limit in seconds")
    args = parser.parse_args()

    if args.time_limit <= 0:
        logger.warning(
            "Invalid time_limit=%d s was requested, falling back to the default value of 30 s",
            args.time_limit,
        )
        args.time_limit = 30

    logger.debug(
        "Command line arguments parsed: image_dir=%s, annotation_path=%s, time_limit=%d s",
        args.file_path, args.annotation_path, args.time_limit,
    )
    return args.file_path, args.annotation_path, args.time_limit


def download_images(filename_images: str, time_limit: int) -> float:
    """Download images for KEYWORD until TARGET_COUNT files or time_limit is reached."""
    logger.info(
        "Image download started: keyword='%s', target_count=%d, time_limit=%d s, image_dir=%s",
        KEYWORD, TARGET_COUNT, time_limit, os.path.abspath(filename_images),
    )

    try:
        if not os.path.isdir(filename_images):
            os.makedirs(filename_images, exist_ok=True)
            logger.info("Image directory created: %s", os.path.abspath(filename_images))
        else:
            logger.debug("Image directory already exists: %s", os.path.abspath(filename_images))
    except OSError:
        logger.exception("Cannot create image directory %s, download aborted", filename_images)
        raise

    start_time = time.monotonic()
    attempt = 0
    previous_count = len(os.listdir(filename_images))

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit:
            logger.warning(
                "Time limit of %d s exhausted after %.2f s and %d crawler run(s); "
                "%d of %d images collected",
                time_limit, elapsed, attempt, previous_count, TARGET_COUNT,
            )
            break
        if previous_count >= TARGET_COUNT:
            logger.info(
                "Target of %d images reached (%d files in %s) after %d crawler run(s)",
                TARGET_COUNT, previous_count, filename_images, attempt,
            )
            break

        attempt += 1
        logger.debug(
            "Crawler run #%d started: keyword='%s', max_num=%d, files_before=%d, time_left=%.2f s",
            attempt, KEYWORD, BATCH_SIZE, previous_count, time_limit - elapsed,
        )
        batch_start = time.monotonic()
        try:
            crawler = BingImageCrawler(storage={"root_dir": filename_images})
            crawler.crawl(keyword=KEYWORD, max_num=BATCH_SIZE)
        except Exception:
            logger.exception(
                "Crawler run #%d for keyword '%s' failed, retrying with the remaining time budget",
                attempt, KEYWORD,
            )
            time.sleep(1)
            continue

        batch_duration = time.monotonic() - batch_start
        current_count = len(os.listdir(filename_images))
        logger.info(
            "Crawler run #%d finished in %.2f s: %d new file(s), %d file(s) total in %s",
            attempt, batch_duration, current_count - previous_count, current_count, filename_images,
        )
        if batch_duration > SLOW_BATCH_SECONDS:
            logger.warning(
                "Crawler run #%d took %.2f s (threshold %.1f s), the search engine may be throttling us",
                attempt, batch_duration, SLOW_BATCH_SECONDS,
            )
        if current_count == previous_count:
            logger.warning(
                "Crawler run #%d returned no new images for keyword '%s', results are probably exhausted",
                attempt, KEYWORD,
            )
        previous_count = current_count
        time.sleep(1)

    elapsed_time = time.monotonic() - start_time
    logger.info(
        "Image download finished in %.2f s: %d file(s) available in %s",
        elapsed_time, previous_count, filename_images,
    )
    return elapsed_time


def make_annotation_file(filename_annotation: str, filename_images: str) -> int:
    """Write absolute and relative paths of every downloaded image into a CSV file."""
    logger.info(
        "Annotation generation started: source_dir=%s, annotation_path=%s",
        filename_images, filename_annotation,
    )
    rows_written = 0
    try:
        parent = os.path.dirname(os.path.abspath(filename_annotation))
        os.makedirs(parent, exist_ok=True)
        with open(filename_annotation, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Absolute Path", "Relative Path"])
            for name in sorted(os.listdir(filename_images)):
                path = os.path.join(filename_images, name)
                writer.writerow([os.path.abspath(path), os.path.relpath(path)])
                rows_written += 1
    except OSError:
        logger.exception(
            "Failed to write annotation file %s for directory %s",
            filename_annotation, filename_images,
        )
        raise

    logger.info(
        "Annotation file %s written successfully: %d image record(s)",
        os.path.abspath(filename_annotation), rows_written,
    )
    if rows_written == 0:
        logger.warning(
            "Annotation file %s contains only the header: no images were found in %s",
            filename_annotation, filename_images,
        )
    return rows_written


class PathIterator:
    """Iterate over [absolute_path, relative_path] pairs from a CSV file or a directory."""

    def __init__(self, source: str) -> None:
        self.items: List[List[str]] = []
        self.counter = 0
        self.source = source

        if os.path.isfile(source):
            logger.debug("Reading image paths from annotation file %s", source)
            try:
                with open(source, newline="", encoding="utf-8") as file:
                    reader = csv.reader(file)
                    next(reader, None)
                    self.items = [row for row in reader if row]
            except (OSError, csv.Error):
                logger.exception("Cannot read annotation file %s, iteration will yield nothing", source)
        elif os.path.isdir(source):
            logger.debug("Reading image paths directly from directory %s", source)
            for name in sorted(os.listdir(source)):
                path = os.path.join(source, name)
                self.items.append([os.path.abspath(path), os.path.relpath(path)])
        else:
            logger.error(
                "Path iterator source %s is neither an existing file nor a directory, no paths to iterate",
                source,
            )

        logger.debug("Path iterator initialised for %s with %d item(s)", source, len(self.items))

    def __iter__(self) -> "PathIterator":
        return self

    def __next__(self) -> List[str]:
        if self.counter >= len(self.items):
            logger.debug(
                "Path iterator over %s exhausted after %d item(s)", self.source, self.counter
            )
            raise StopIteration
        path = self.items[self.counter]
        self.counter += 1
        return path


def main() -> int:
    setup_logging()
    logger.info("Pig image dataset script started (pid=%d)", os.getpid())

    filename_images, filename_annotation, time_limit = parsing()

    try:
        elapsed_time = download_images(filename_images, time_limit)
        records = make_annotation_file(filename_annotation, filename_images)
    except OSError:
        logger.critical(
            "Dataset creation aborted because of a filesystem error; see the traceback above",
        )
        return 1
    except Exception:
        logger.critical("Dataset creation aborted by an unexpected error", exc_info=True)
        return 1

    final_count = len(os.listdir(filename_images))
    logger.info(
        "Run summary: %d image(s) in %s, %d annotation record(s), elapsed time %.2f s",
        final_count, filename_images, records, elapsed_time,
    )
    if final_count < TARGET_COUNT:
        logger.warning(
            "Dataset is incomplete: %d of %d requested images were downloaded within %d s",
            final_count, TARGET_COUNT, time_limit,
        )

    preview_limit = 3
    checked = 0
    for index, path in enumerate(PathIterator(filename_annotation), start=1):
        checked += 1
        if index <= preview_limit:
            logger.debug(
                "Annotation record #%d verified: absolute_path=%s, relative_path=%s",
                index, path[0], path[1],
            )
    logger.info(
        "Annotation self-check finished: %d record(s) iterated from %s, first %d logged at DEBUG level",
        checked, filename_annotation, min(preview_limit, checked),
    )

    logger.info("Pig image dataset script finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())