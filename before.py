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
        
     def rotate(self, degrees):
        self.image = self.image.rotate(degrees, expand=True)
        print(f"Rotated by {degrees} degrees")

    def flip_horizontal(self):
        self.image = self.image.transpose(Image.FLIP_LEFT_RIGHT)
        print("Flipped horizontally")

    def flip_vertical(self):
        self.image = self.image.transpose(Image.FLIP_TOP_BOTTOM)
        print("Flipped vertically")

    def crop(self, left, top, right, bottom):
        self.image = self.image.crop((left, top, right, bottom))
        print(f"Cropped to: {left},{top}-{right},{bottom}")

    def convert_to_grayscale(self):
        self.image = self.image.convert('L')
        print("Converted to grayscale")

    def adjust_brightness(self, factor):
        enhancer = ImageEnhance.Brightness(self.image)
        self.image = enhancer.enhance(factor)
        print(f"Adjusted brightness by factor: {factor}")

    def adjust_contrast(self, factor):
        enhancer = ImageEnhance.Contrast(self.image)
        self.image = enhancer.enhance(factor)
        print(f"Adjusted contrast by factor: {factor}")

    def adjust_sharpness(self, factor):
        enhancer = ImageEnhance.Sharpness(self.image)
        self.image = enhancer.enhance(factor)
        print(f"Adjusted sharpness by factor: {factor}")

    def apply_blur(self, radius=2):
        self.image = self.image.filter(ImageFilter.GaussianBlur(radius))
        print(f"Applied blur with radius: {radius}")

    def apply_sharpen(self):
        self.image = self.image.filter(ImageFilter.SHARPEN)
        print("Applied sharpen filter")

    def apply_edge_enhance(self):
        self.image = self.image.filter(ImageFilter.EDGE_ENHANCE)
        print("Applied edge enhance filter")

