def text_encryption(data: str, key: str, alphabet: str) -> str:
    """
    Encrypt input text using a complex Caesar cipher with a key.

    :param data: Text to encrypt.
    :param key: Encryption key.
    :param alphabet: Alphabet used for encryption.
    :return: Encrypted text.
    """
    try:
        data, key = data.lower(), key.lower()
        extended_key = (key * (len(data) // len(key) + 1))[: len(data)]
        encrypted_text = ""

        for symbol, shift in zip(data, extended_key):
            if symbol in alphabet and shift in alphabet:
                new_position = (alphabet.find(symbol) + alphabet.find(shift)) % len(
                    alphabet
                )
                encrypted_text += alphabet[new_position]
            else:
                encrypted_text += symbol

        return encrypted_text
    except Exception as e:
        print(f"An error occurred during encryption1: {e}")
        return ""


def text_decryption(encrypted_data: str, key: str, alphabet: str) -> str:
    """
    Decrypt input text encrypted with a complex Caesar cipher using a key.

    :param encrypted_data: Text to decrypt.
    :param key: Encryption key used during encryption.
    :param alphabet: Alphabet used for encryption.
    :return: Decrypted text.
    """
    if not encrypted_data or not key or not alphabet:
        raise ValueError("Encrypted data, key, and alphabet must not be empty.")

    encrypted_data, key = encrypted_data.lower(), key.lower()
    extended_key = (key * (len(encrypted_data) // len(key) + 1))[: len(encrypted_data)]
    decrypted_text = ""

    for symbol, shift in zip(encrypted_data, extended_key):
        if symbol in alphabet and shift in alphabet:
            new_position = (alphabet.find(symbol) - alphabet.find(shift)) % len(
                alphabet
            )
            decrypted_text += alphabet[new_position]
        else:
            decrypted_text += symbol

    return decrypted_text


def calculate_symbol_frequency(data: str) -> dict[str, float]:
    """
    Calculate symbol frequency in the given text.

    :param data: Input text.
    :return: Dictionary of symbols and their frequencies.
    """
    try:

        result = {}

        for symbol in data:
            if symbol == "\n":
                continue
            result[symbol] = result.get(symbol, 0) + 1

        for symbol, count in result.items():
            result[symbol] = count / len(data)

        return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    except Exception as e:
        print(f"An error occurred while calculating symbol frequency: {e}")
        return {}


def make_key(freq_alp: dict[str, float], freq_task: dict[str, float]) -> dict[str, str]:
    """
    Create a mapping of symbols based on their frequencies.

    :param freq_alp: Frequencies of alphabet symbols.
    :param freq_task: Frequencies of task symbols.
    :return: Mapping of task symbols to alphabet symbols.
    """
    result = {}

    for alp_symb, task_symb in zip(freq_alp.keys(), freq_task.keys()):
        result[task_symb] = alp_symb

    return result


def decryption_cod3(data: str, key: dict[str, str]) -> str:
    """
    Replace symbols in the data using the provided key.

    :param data: Input text.
    :param key: Dictionary mapping symbols to replacements.
    :return: Modified text.
    """
    result = ""
    for symbol in data:
        result += key.get(symbol, symbol)
    return result
