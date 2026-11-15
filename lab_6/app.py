import argparse
from fileworks import *
from assymetric import *
from symmetric import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json_file", type=str, default="settings.json", help="Json file with keys, modules and texts paths")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-g", "--generation", action="store_true", help="Start key generation")
    group.add_argument("-e", "--encryption", action="store_true", help="Start encryption")
    group.add_argument("-d", "--decryption", action="store_true", help="Start encryption")
    
    parser.add_argument("-sy", "--symmetric_key", help="Path to user's symmetric key")

    args = parser.parse_args()
    
    try:
        paths = read_json(args.json_file)
        symmetric_key_file = args.symmetric_key if args.symmetric_key else paths["symmetric_key"]
        secret_key_file = paths["secret_key"]
        public_key_file = paths["public_key"]
        initial_file = paths["initial_file"]
        encrypted_file = paths["encrypted_file"]
        decrypted_file = paths["decrypted_file"]
    except Exception as e:
        print(f"Error: {e}")
        raise e

    match args:
        case _ if args.generation:
            try:
                try:
                    symmetric_key = read_bytes(symmetric_key_file)
                except FileNotFoundError:
                    symmetric_key = symmetric_key_generation()
                asymmetric_keys = asymmetric_keys_generation()
                asymmetric_keys_writter(secret_key_file, public_key_file, asymmetric_keys)
                encrypted_key = key_encription(symmetric_key, asymmetric_keys[1])
                symmetric_key_writter(symmetric_key_file, encrypted_key)
            except Exception as e:
                print(f"Error: {e}")

        case _ if args.encryption:
            try:
                key = read_bytes(symmetric_key_file)
                secret_key = get_secret_key(secret_key_file)
                key = decrypt_key(key, secret_key)

                text = read_str(initial_file)
                result_text, nonce = text_encryption(text, key)
                encrypted_ChaCha20_writer(result_text, nonce, encrypted_file)
            except Exception as e:
                print(f"Error {e}")

        case _ if args.decryption:
            try: 
                key = read_bytes(symmetric_key_file)
                secret_key = get_secret_key(secret_key_file)
                key = decrypt_key(key, secret_key)

                text = read_bytes(encrypted_file)
                result_text = text_decryption(text, key)
                str_writer(result_text, decrypted_file)
            except Exception as e:
                print(f"Error {e}")

if __name__ == "__main__":
    main()