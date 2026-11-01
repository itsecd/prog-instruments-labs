import random
import time

def write_to_file(filename : str, num : str) -> None:
    with open(filename, "w", encoding="utf-8") as file:
        file.write(num)
    

def generator(filename : str) -> None:
    random.seed(time.time())
    text = ""
    for i in range(128):
        random_int = random.randint(0, 1)
        text += str(random_int)
    write_to_file(filename , text)


def main() -> None:

    filename = "output_py.txt"
    generator(filename)

if __name__ == "__main__":
    main()