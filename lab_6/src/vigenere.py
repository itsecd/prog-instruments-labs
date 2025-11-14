from const import ALPHABET


def get_key_symb(key: str, index: int) -> str:
    """
    Returns the character from the key based on the index.

    :param key: The original key.
    :param index: The index of the character in the text.
    :return: The corresponding character from the key.
    """
    if not key:
        raise ValueError("Key can't be empty!")

    return key[index % len(key)]


def get_encrypted_symb(old_sym: str, key_sym: str) -> str:
    """
    Encrypts a single character using the Vigenère cipher.

    :param old_sym: The original text character.
    :param key_sym: The corresponding key character.
    :return: The encrypted character.
    """

    if old_sym.isalpha():
        try:

            current_idx = ALPHABET.index(old_sym.lower())
            key_idx = ALPHABET.index(key_sym.lower())

        except:

            raise ValueError(f"Character '{old_sym}' or '{key_sym}' not found in alphabet.")

        if current_idx + key_idx >= len(ALPHABET):

            encrypt_idx =  current_idx + key_idx - len(ALPHABET)

        else:

            encrypt_idx = current_idx + key_idx

        if old_sym == old_sym.upper():

            return ALPHABET[encrypt_idx].upper()

        return ALPHABET[encrypt_idx]

    else:

        return old_sym


def get_decrypted_symb(encrypted_sym: str, key_sym: str) -> str:
    """
    Decrypts a single character using the Vigenère cipher.

    :param encrypted_sym: The encrypted character.
    :param key_sym: The corresponding key character.
    :return: The decrypted character.
    """

    if encrypted_sym.isalpha():

        try:

            encrypted_idx = ALPHABET.index(encrypted_sym.lower())
            key_idx = ALPHABET.index(key_sym.lower())

        except:

            raise ValueError(f"Character '{encrypted_sym}' or '{key_sym}' not found in alphabet.")

        if encrypted_idx - key_idx < 0:

            decrypted_idx = encrypted_idx - key_idx + len(ALPHABET)

        else:

            decrypted_idx = encrypted_idx - key_idx

        if encrypted_sym == encrypted_sym.upper():

            return ALPHABET[decrypted_idx].upper()

        return ALPHABET[decrypted_idx]

    else:

        return encrypted_sym


def vigenere_cipher_encrypt(input_text: str, key: str) -> str:
    """
    Encrypts text.

    :param input_text: The original text to be encrypted.
    :param key: The encryption key.
    :return: The encrypted text.
    """

    if not input_text:

        raise ValueError("Input text can't be empty")

    if not key:

        raise ValueError("Key can't be empty")

    encrypted_text = ""

    for i in range(len(input_text)):

        text_sym = input_text[i]
        key_sym = get_key_symb(key, i)

        encrypted_text += get_encrypted_symb(text_sym, key_sym)

    return encrypted_text

def vigenere_cipher_decrypt(encrypted_text: str, key: str) -> str:
    """
    Decrypts text.

    :param encrypted_text: The encrypted text.
    :param key: The encryption key.
    :return: The decrypted text.
    """

    if not encrypted_text:

        raise ValueError("Encrypted text can't be empty")

    if not key:

        raise ValueError("Key can't be empty")

    decrypted_text = ""

    for i in range(len(encrypted_text)):

        encrypted_sym = encrypted_text[i]
        key_sym = get_key_symb(key, i)

        decrypted_text += get_decrypted_symb(encrypted_sym, key_sym)

    return decrypted_text
