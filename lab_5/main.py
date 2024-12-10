import argparse

from asymetric import Asymmetric
from symmetric import Symmetric
from file_readers import read_json_file, write_file, read_txt, write_key


def generate_keys(sym: Symmetric, asym: Asymmetric, sym_key_path: str, public_key_path: str, private_key_path: str) -> None:
    """This function generates symmetric and asymmetric keys and serializes them"""
    sym.generate_key()
    asym.generate_key()
    asym.serialize_public_key(public_key_path)
    asym.serialize_private_key(private_key_path)
    sym.serialize_key(sym_key_path)


def encrypt_sym_key(sym: Symmetric, asym: Asymmetric, sym_key_path: str, public_key_path: str, path: str) -> None:
    """This function encrypts key using asymmetric encryption and saves it"""
    sym.deserialize_key(sym_key_path)
    asym.deserialize_public_key(public_key_path)
    encrypted_key = asym.encrypt_key(sym.key)
    write_key(path, encrypted_key)


def decrypt_sym_key(sym:Symmetric, asym: Asymmetric, enc_sym_key_path: str, private_key_path: str, path: str) -> None:
    """This function decrypts key using asymtreic decryption and saves it"""
    asym.deserialize_private_key(private_key_path)
    sym.deserialize_key(enc_sym_key_path)
    decrypted_key = asym.decrypt_key(sym.key)
    write_key(path, decrypted_key)

def encrypt_text(sym:Symmetric, text_path: str, decrypted_key_path: str, path: str) -> None:
    """This function encrypts text using symmetric encryption and saves it to a file"""
    text = read_txt(text_path)
    sym.deserialize_key(decrypted_key_path)
    encrypted_text = sym.encrypt_text(text)
    write_file(path, encrypted_text)


def decrypt_text(sym: Symmetric, encrypted_text: str, decrypted_key_path: str, path: str) -> None:
    """This function decrypts text using symmetric decryption and save it to a file"""
    text = read_txt(encrypted_text)
    sym.deserialize_key(decrypted_key_path)
    decrypted_text = sym.decrypt_text(text)
    write_file(path, decrypted_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument('-gen', '--generation', help = 'Starts key generation text mode')
    group.add_argument('-enc_key', '--encryption_key', help = 'Starts encryption key mode')
    group.add_argument('-dec_key', '--decryption_key', help = 'Starts decryption key mode')
    group.add_argument('-enc', '--encryption', help = 'Starts encryption text mode')
    group.add_argument('-dec', '--decryption', help = 'Starts decryption text mode')
    parser.add_argument("settings", type=str, help = "Path to the json file with the settings")

    args = parser.parse_args()
    settings = read_json_file(args.settings)
    symmetric = Symmetric()
    asymmetric = Asymmetric()
    match args:
        case args if args.generation:
            generate_keys(symmetric, asymmetric, settings["symmetric_key"], settings["public_key"], settings["private_key"])
        case args if  args.encryption_key:
            encrypt_sym_key(symmetric, asymmetric, settings["symmetric_key"], settings["public_key"], settings["enc_sym_key"])
        case args if args.decryption_key:
            decrypt_sym_key(symmetric, asymmetric, settings["enc_sym_key"], settings["private_key"], settings["dec_sym_key"])
        case args if  args.encryption:
            encrypt_text(symmetric, settings["initial_file"], settings["dec_sym_key"], settings["encrypted_file"])
        case args if args.decryption:
            decrypt_text(symmetric, settings["encrypted_file"], settings["dec_sym_key"], settings["decrypted_file"])