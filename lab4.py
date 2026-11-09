import argparse

ALPHABET = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ "

def decryption(text, key):
    table = []
    for i in range(len(ALPHABET)):
        row = ALPHABET[i:] + ALPHABET[:i]
        table.append(row)

    decrypted_text = ""
    key_counter = 0
    for i in range(len(text)):
        char = text[i]
        char_upper = char.upper()

        if char_upper in ALPHABET:
            key_char = key[key_counter % len(key)]
            key_index = ALPHABET.index(key_char.upper())

            encode_char = ""
            for j in range(len(table)):
                if char_upper == table[j][key_index]:
                    encode_char = table[j][0]
            decrypted_text += encode_char
            key_counter += 1
    return decrypted_text

def encode(text, key):
    table = []
    for i in range(len(ALPHABET)):
        row = ALPHABET[i:] + ALPHABET[:i]
        table.append(row)

    encode_text = ""
    key_counter = 0
    for i in range(len(text)):
        char = text[i]
        char_upper = char.upper()

        if char_upper in ALPHABET:
            key_char = key[key_counter % len(key)]
            key_index = ALPHABET.index(key_char.upper())

            char_index = ALPHABET.index(char_upper)
            encode_char = table[char_index][key_index]
            encode_text += encode_char
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
