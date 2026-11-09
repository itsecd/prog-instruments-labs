import argparse

ALPHABET = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ "

def generate_vigenere_table():
    return [ALPHABET[i:] + ALPHABET[:i] for i in range(len(ALPHABET))]

def get_vigenere_char(char, key_char, table, mode="encode"):
    char_upper = char.upper()
    if char_upper not in ALPHABET:
        return ""

    key_index = ALPHABET.index(key_char.upper())
    
    if mode == "encode":
        char_index = ALPHABET.index(char_upper)
        return table[char_index][key_index]
    elif mode == "decode":
        for row in range(len(table)):
            if char_upper == table[row][key_index]:
                return table[row][0]
    return ""

def decryption(text, key):
    table = generate_vigenere_table()
    decrypted_text = ""
    key_counter = 0
    for char in text:
        if char.upper() in ALPHABET:
            key_char = key[key_counter % len(key)]
            decrypted_text += get_vigenere_char(char, key_char, table, mode="decode")
            key_counter += 1
    return decrypted_text

def encode(text, key):
    table = generate_vigenere_table()
    encode_text = ""
    key_counter = 0
    for char in text:
        if char.upper() in ALPHABET:
            key_char = key[key_counter % len(key)]
            encode_text += get_vigenere_char(char, key_char, table, mode="encode")
            key_counter += 1
    return encode_text

def read_file(filename, strip_whitespace=False):
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
        return content.strip() if strip_whitespace else content
    
def write_result(encode_text, filenameoutput):
    with open(filenameoutput, "w", encoding="utf-8") as file:
        file.write(encode_text)

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("filenametext", type=str)
    parser.add_argument("filenamekey", type=str)
    parser.add_argument("filenametextoutput", type=str)
    args = parser.parse_args()
    return args.filenametext, args.filenamekey, args.filenametextoutput

def main():
    filenametext, filenamekey, filenametextoutput = parsing()
    key = read_file(filenamekey, strip_whitespace=True)
    text = read_file(filenametext)
    encode_text = encode(text, key)
    decryption(encode_text, key)
    write_result(encode_text, filenametextoutput)

if __name__ == "__main__":
    main()
