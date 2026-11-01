import argparse
import csv
import os
from typing import Iterator

from icrawler.builtin import BingImageCrawler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download images and create annotation CSV"
    )
    parser.add_argument(
        "file_save", type=str, help="Path to directory for saving photos"
    )
    parser.add_argument(
        "annotation", type=str, help="Path to annotation CSV file"
    )
    parser.add_argument(
        "number", type=int, help="Number of images (from 50 to 1000)"
    )
    parser.add_argument(
        "keywords", nargs="+", help="Keywords for image search"
    )
    args = parser.parse_args()

    if not (50 <= args.number <= 1000):
        raise ValueError("Number of images must be between 50 and 1000")

    return args


def download_images(
    keywords: list[str], save_dir: str, number: int
) -> list[dict[str, str]]:
    data: list[dict[str, str]] = []

    for keyword in keywords:
        folder = os.path.join(save_dir, keyword)
        crawler = BingImageCrawler(storage={"root_dir": folder})
        crawler.crawl(keyword=keyword, max_num=number)

        for root, _, files in os.walk(folder):
            for filename in files:
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                    abs_path = os.path.abspath(os.path.join(root, filename))
                    rel_path = os.path.relpath(abs_path, save_dir)
                    data.append(
                        {
                            "absolute_path": abs_path,
                            "relative_path": rel_path,
                            "keyword": keyword,
                        }
                    )

    return data


class AnnotationIterator:
    def __init__(self, annotation_path: str) -> None:
        self.annotation_path = annotation_path
        self.data: list[dict[str, str]] = []
        self.index = 0

        with open(
            annotation_path, mode="r", encoding="utf-8", newline=""
        ) as file:
            reader = csv.DictReader(file)
            self.data = list(reader)

    def __iter__(self) -> Iterator[dict[str, str]]:
        return self

    def __next__(self) -> dict[str, str]:
        if self.index >= len(self.data):
            raise StopIteration
        value = self.data[self.index]
        self.index += 1
        return value


def write_annotation(
    annotation_path: str, data: list[dict[str, str]]
) -> None:
    fieldnames = ["absolute_path", "relative_path", "keyword"]

    with open(
        annotation_path, mode="w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def main() -> None:
    args = parse_args()
    data = download_images(args.keywords, args.file_save, args.number)
    write_annotation(args.annotation, data)

    iterator = AnnotationIterator(args.annotation)
    for row in iterator:
        print(row)
        print()


if __name__ == "__main__":
    main()