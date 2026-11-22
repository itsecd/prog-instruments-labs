import requests
from func import get_args, run_downloads, print_summary
import logging
from log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def main(out_dir: str, csv_out: str) -> None:
    genres = ["rock", "jazz", "R&B"]
    logger.info("Application started: out_dir=%s csv=%s", out_dir, csv_out)
    stats = run_downloads(out_dir, csv_out, genres)
    print_summary(csv_out, stats)
    logger.info("Application finished: tracks=%d", stats["tracks_total"])

if __name__ == "__main__":
    args = get_args()
    try:
        main(args.output, args.csv)
    except FileNotFoundError as err:
        logger.error("File not found: %s", err.filename)
    except requests.RequestException as err:
        logger.error("Network error: %s", err)
    except Exception:
        logger.critical("Unexpected error", exc_info=True)