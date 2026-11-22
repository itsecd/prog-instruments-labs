import argparse
from fileworks import *
from assymetric import *
from symmetric import *


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-g", "--generation", action="store_true", help="Start key generation")
    group.add_argument("-e", "--encryption", action="store_true", help="Start encryption")
    group.add_argument("-d", "--decryption", action="store_true", help="Start decryption")
    parser.add_argument("-sy", "--symmetric_key", help="Path to user's symmetric key")

    parser.add_argument("-c", "--config", default="lab_9/configs/config.yaml", help="Path to config") 
    parser.add_argument("-env", "--environment", choices=["lab_9/configs/dev.yaml", "lab_9/configs/prod.yaml"],
                        default="lab_9/configs/dev.yaml", help="Environment: dev or prod (default: dev)")

    args = parser.parse_args()

    try:
        cfg = load_config(args.config, args.environment)
    except Exception as e:
        print(f"Error loading config: {e}")
        raise e

    symmetric_key_file = args.symmetric_key if args.symmetric_key else cfg.files.symmetric_key
    secret_key_file = cfg.files.secret_key
    public_key_file = cfg.files.public_key
    initial_file = cfg.files.initial_file
    encrypted_file = cfg.files.encrypted_file
    decrypted_file = cfg.files.decrypted_file

    sym_key_size = cfg.crypto.symmetric.key_size
    nonce_size = cfg.crypto.symmetric.nonce_size
    rsa_key_size = cfg.crypto.asymmetric.key_size
    rsa_public_exponent = cfg.crypto.asymmetric.public_exponent

    match args:
        case _ if args.generation:
            try:
                try:
                    symmetric_key = read_bytes(symmetric_key_file)
                except FileNotFoundError:
                    symmetric_key = symmetric_key_generation(key_size=sym_key_size)

                asymmetric_keys = asymmetric_keys_generation(
                    key_size=rsa_key_size,
                    public_exponent=rsa_public_exponent,
                )
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
                result_text, nonce = text_encryption(text, key, nonce_size)
                encrypted_ChaCha20_writer(result_text, nonce, encrypted_file)
            except Exception as e:
                print(f"Error: {e}")

        case _ if args.decryption:
            try:
                key = read_bytes(symmetric_key_file)
                secret_key = get_secret_key(secret_key_file)
                key = decrypt_key(key, secret_key)

                text = read_bytes(encrypted_file)
                result_text = text_decryption(text, key, sym_key_size, nonce_size)
                str_writer(result_text, decrypted_file)
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()