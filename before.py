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

    def add_watermark(self, text, position=(10, 10), font_size=20, opacity=128):
        watermark = self.image.copy()
        draw = ImageDraw.Draw(watermark)
    
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
    
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    
        if isinstance(position, str):
            if position == "center":
                x = (self.image.width - text_width) // 2
                y = (self.image.height - text_height) // 2
                position = (x, y)
            elif position == "bottom-right":
                x = self.image.width - text_width - 10
                y = self.image.height - text_height - 10
                position = (x, y)
    
        padding = 5
        draw.rectangle(
            [
                position[0] - padding,
                position[1] - padding,
                position[0] + text_width + padding,
                position[1] + text_height + padding
            ],
            fill=(0, 0, 0, opacity // 2)
        )
    
        draw.text(position, text, font=font, fill=(255, 255, 255, opacity))
    
        self.image = Image.alpha_composite(
            self.image.convert('RGBA'), watermark
        )
        print(f"Added watermark: '{text}'")

        def add_border(self, thickness=10, color=(255, 255, 255)):
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')

        new_width = self.image.width + 2 * thickness
        new_height = self.image.height + 2 * thickness

        new_image = Image.new('RGB', (new_width, new_height), color)
        new_image.paste(self.image, (thickness, thickness))
        self.image = new_image
        print(f"Added border: {thickness}px {color}")

    def create_thumbnail(self, size=(128, 128)):
        self.image.thumbnail(size, Image.Resampling.LANCZOS)
        self.output_path = (
            self.input_path.parent /
            f"{self.input_path.stem}_thumb{self.input_path.suffix}"
        )
        print(f"Created thumbnail: {size[0]}x{size[1]}")

    def get_image_info(self):
        info = {
            'format': self.image.format,
            'mode': self.image.mode,
            'size': self.image.size,
            'width': self.image.width,
            'height': self.image.height
        }
        return info

    def print_info(self):
    info = self.get_image_info()
    print("\n=== Image Information ===")
    for key, value in info.items():
        print(f"{key}: {value}")


def process_single_image(args):
    processor = ImageProcessor(args.input, args.output)

    if args.resize:
        if 'x' in args.resize:
            width, height = map(int, args.resize.split('x'))
            processor.resize(width=width, height=height)
        elif args.resize.endswith('%'):
            percentage = int(args.resize[:-1])
            processor.resize(percentage=percentage)
        else:
            width = int(args.resize)
            processor.resize(width=width)

    if args.rotate:
        processor.rotate(args.rotate)

    if args.flip == 'horizontal':
        processor.flip_horizontal()
    elif args.flip == 'vertical':
        processor.flip_vertical()

    if args.crop:
        left, top, right, bottom = map(int, args.crop.split(','))
        processor.crop(left, top, right, bottom)

    if args.grayscale:
        processor.convert_to_grayscale()

    if args.brightness:
        processor.adjust_brightness(args.brightness)

    if args.contrast:
        processor.adjust_contrast(args.contrast)

    if args.sharpness:
        processor.adjust_sharpness(args.sharpness)

    if args.blur:
        processor.apply_blur(args.blur)

    if args.sharpen:
        processor.apply_sharpen()

    if args.edge_enhance:
        processor.apply_edge_enhance()

    if args.watermark:
        processor.add_watermark(args.watermark)

    if args.border:
        thickness, color = args.border.split(',')
        thickness = int(thickness)
        if color.startswith('#'):
            color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        else:
            color = tuple(map(int, color.split(':')))
        processor.add_border(thickness, color)

    if args.thumbnail:
        size = tuple(map(int, args.thumbnail.split('x')))
        processor.create_thumbnail(size)

    if args.info:
        processor.print_info()

    if not args.info:
        processor.save_image(quality=args.quality)


