from PIL import Image
import os


# Создаем простые тестовые изображения
def create_test_images():
    # Создаем папку если её нет
    os.makedirs('tests/test_images', exist_ok=True)

    # Создаем маленькое красное изображение 10x10
    img1 = Image.new('RGB', (10, 10), color='red')
    img1.save('tests/test_images/test_red.png')

    # Создаем маленькое синее изображение 10x10
    img2 = Image.new('RGB', (10, 10), color='blue')
    img2.save('tests/test_images/test_blue.jpg')

    # Создаем маленькое зеленое изображение 10x10
    img3 = Image.new('RGB', (10, 10), color='green')
    img3.save('tests/test_images/test_green.gif')

    print("✅ Тестовые изображения созданы!")


if __name__ == "__main__":
    create_test_images()