import argparse
from typing import Any

from AES import decrypt as aes_decrypt
from AES import encrypt as aes_encrypt
from key_gen import generate_keys_pipeline
from RSA import decrypt as rsa_decrypt
from RSA import load_private_key
from utils import load_config, read_bytes, write_bytes


def decrypt_symmetric_key(secret_key_path: str, enc_key_path: str) -> bytes:
    """
    Расшифровывает симметричный AES ключ с помощью RSA.
    """
    private_key = load_private_key(secret_key_path)
    encrypted_key = read_bytes(enc_key_path)
    return rsa_decrypt(encrypted_key, private_key)


def encrypt_file(config: dict[str, Any]) -> None:
    """
    Шифрует файл с использованием гибридной схемы RSA + AES.
    """
    print("[*] Encryption started")

    sym_key = decrypt_symmetric_key(
        config["secret_key"],
        config["symmetric_key"],
    )

    data = read_bytes(config["initial_file"])
    encrypted_data = aes_encrypt(sym_key, data)

    write_bytes(config["encrypted_file"], encrypted_data)

    print("[+] Encryption finished")


def decrypt_file(config: dict[str, Any]) -> None:
    """
    Расшифровывает файл с использованием гибридной схемы RSA + AES.
    """
    print("[*] Decryption started")

    sym_key = decrypt_symmetric_key(
        config["secret_key"],
        config["symmetric_key"],
    )

    encrypted_data = read_bytes(config["encrypted_file"])
    decrypted_data = aes_decrypt(sym_key, encrypted_data)

    write_bytes(config["decrypted_file"], decrypted_data)

    print("[+] Decryption finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid RSA + AES crypto system")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"{args.config}")

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"[-] Failed to load config: {e}")
        return

    while True:
        print("\n------MENU------")
        print("1. Generate keys")
        print("2. Encrypt file (using existing keys)")
        print("3. Decrypt file (using existing keys)")
        print("4. Use custom keys (update config paths)")
        print("5. Exit")

        choice = input("Select option (1-5): ").strip()

        match choice:
            case "1":
                try:
                    print("[*] Key generation")
                    generate_keys_pipeline(config)
                except Exception as e:
                    print(f"[-] Error during key generation: {e}")

            case "2":
                try:
                    print("[*] File encryption")
                    encrypt_file(config)
                except FileNotFoundError as e:
                    print(f"[-] File not found: {e}")
                except PermissionError as e:
                    print(f"[-] Permission denied: {e}")
                except Exception as e:
                    print(f"[-] Error during encryption: {e}")

            case "3":
                try:
                    print("[*] File decryption")
                    decrypt_file(config)
                except FileNotFoundError as e:
                    print(f"[-] File not found: {e}")
                except PermissionError as e:
                    print(f"[-] Permission denied: {e}")
                except Exception as e:
                    print(f"[-] Error during decryption: {e}")

            case "4":
                try:
                    print("[*] Use custom keys")
                    custom_private = input("Enter path to your private key: ").strip()
                    custom_public = input("Enter path to your public key: ").strip()
                    custom_sym = input("Enter path to your encrypted symmetric key: ").strip()

                    if custom_private:
                        config["secret_key"] = custom_private
                    if custom_public:
                        config["public_key"] = custom_public
                    if custom_sym:
                        config["symmetric_key"] = custom_sym
                    print("[+] Custom keys paths updated")
                except Exception as e:
                    print(f"[-] Error updating custom keys: {e}")

            case "5":
                print("[+] Exiting")
                break

            case _:
                print("[-] Invalid option. Please select 1-5")


if __name__ == "__main__":
    main()
