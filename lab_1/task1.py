import sys

from ordered_set import OrderedSet

from values import ALLOWED_SYMBOLS, CHECK_RESULT, RUSSIAN_ALPHBET


def read_text(filename: str) -> str:
    """
    Docstring for read_text

    :param filename: Input filename for reading
    :type filename: str
    :return: Text from file
    :rtype: str
    """
    try:
        with open(filename, encoding="utf-8") as f:
            return f.read()

    except FileExistsError:
        print(f"Failed to open file {filename}")
        exit(2)
    except PermissionError:
        print(f"Permission denied for {filename}")
        exit(2)
    except Exception as e:
        print(e)
        exit(2)


def get_alphabets(key: str) -> tuple[list[str], list[str]]:
    """
    Docstring for get_alphabets

    :param key: Seceret key
    :type key: str
    :return: Source and encoding alphabets for this key
    :rtype: str
    """
    encode_alphabet = OrderedSet(key.lower())

    for i in ALLOWED_SYMBOLS:
        if len(encode_alphabet) < len(RUSSIAN_ALPHBET):
            encode_alphabet.add(i)

    source_alphabet = list(RUSSIAN_ALPHBET.lower())
    encode_alphabet = list(encode_alphabet)

    return source_alphabet, encode_alphabet


def encode(source: str, key: str) -> str:
    """
    Docstring for encode

    :param source: Source text to encode
    :type source: str
    :param key: Secret key (English chars + digits only)
    :type key: str
    :return: Encoded text
    :rtype: str
    """
    result = source.lower()

    source_alphabet, encode_alphabet = get_alphabets(key)

    for i in range(len(RUSSIAN_ALPHBET)):
        result = result.replace(source_alphabet[i], encode_alphabet[i])

    return result


def decode(data: str, key: str) -> str:
    """
    Docstring for decode

    :param data: Encoded text to decode
    :type data: str
    :param key: Secret key
    :type key: str
    :return: Decoded text
    :rtype: str
    """
    source_alphabet, encode_alphabet = get_alphabets(key)

    for i in range(len(RUSSIAN_ALPHBET)):
        data = data.replace(encode_alphabet[i], source_alphabet[i])

    return data


def main() -> None:
    """
    Docstring for main
    """
    if len(sys.argv) != 4:
        print("Usage: python task1.py input.txt key.txt output.txt")
        exit(1)

    input_file = sys.argv[1]
    key_file = sys.argv[2]
    output_file = sys.argv[3]

    input_text = read_text(input_file)
    key = read_text(key_file)

    encode_text = encode(input_text, key)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(encode_text)
            print(f"Saved to {output_file}")
    except PermissionError:
        print(f"Permission denied for {output_file}")
        exit(2)
    except Exception as e:
        print(e)
        exit(2)

    if CHECK_RESULT:
        print("Source text: ")
        print(input_text)
        print("===============")
        print("Key: ")
        print(key)
        print("===============")
        print("Decode text: ")
        print(decode(encode_text, key))


if __name__ == "__main__":
    main()
