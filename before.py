import os
import argparse
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import math
from pathlib import Path
import sys


class ImageProcessor:
    def __init__(self, input_path, output_path=None):
        self.input_path = Path(input_path)
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = (
                self.input_path.parent /
                f"{self.input_path.stem}_processed{self.input_path.suffix}"
            )

        self.image = None
        self.load_image()

    def load_image(self):
        try:
            self.image = Image.open(self.input_path)
            print(
                f"Loaded image: {self.input_path} "
                f"({self.image.size[0]}x{self.image.size[1]})"
            )
        except Exception as e:
            print(f"Error loading image: {e}")
            sys.exit(1)

    def save_image(self, quality=95):
        try:
            self.image.save(self.output_path, quality=quality)
            print(f"Saved image to: {self.output_path}")
        except Exception as e:
            print(f"Error saving image: {e}")

    def resize(self, width=None, height=None, percentage=None):
        if percentage:
            new_width = int(self.image.width * percentage / 100)
            new_height = int(self.image.height * percentage / 100)
        elif width and height:
            new_width, new_height = width, height
        elif width:
            ratio = width / self.image.width
            new_height = int(self.image.height * ratio)
            new_width = width
        elif height:
            ratio = height / self.image.height
            new_width = int(self.image.width * ratio)
            new_height = height
        else:
            print("No resize parameters provided")
            return

        self.image = self.image.resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )
        print(f"Resized to: {new_width}x{new_height}")
