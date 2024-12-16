import argparse

from moduls.reader_writer import Texting
from moduls.cryptography import cryptography
from const import WAY, SIZE

if __name__ == "__main__":

    print("start program")
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-gen", "--generation", help="Запускает режим генерации ключей")
    group.add_argument("-enc", "--encryption", help="Запускает режим шифрования")
    group.add_argument("-dec", "--decryption", help="Запускает режим дешифрования")
    args = parser.parse_args()

    ways = Texting.read_json_file(WAY)
    
    match args:
        case args if args.generation:
            cryptography.generation_proc(ways[2], ways[1], ways[0])
        case args if  args.encryption:
            cryptography.encryption_proc(ways[4], ways[3], ways[2], ways[1], ways[0])
        case args if args.decryption:
            cryptography.decryption_proc(ways[5], ways[4], ways[2], ways[1], ways[0])


        
        
