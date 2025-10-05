import argparse

from cryptosystem import CryptoSystem


def parsing():
    parser = argparse.ArgumentParser(description="Hybrid cipher system "
                                                 "(Camellia + RSA)")
    subparsers = parser.add_subparsers(dest='command', help='Available command')

    # parser for generate keys
    keygen_parser = subparsers.add_parser('generate', help='Generate keys')
    keygen_parser.add_argument('-encrypted_sym_key', required=True,
                               help='Path for save encrypted symmetric keys')
    keygen_parser.add_argument('-public_key', required=True,
                               help='Path to save public key')
    keygen_parser.add_argument('-private_key', required=True,
                               help='Path to save private key')
    keygen_parser.add_argument('-key_length', required=True, type=int,
                               help='Symmetric key length (128, 192, 256)')

    # parser for encrypt
    encrypt_parser = subparsers.add_parser('encrypt', help='Encrypt file')
    encrypt_parser.add_argument('-input_file', required=True,
                                help='Path to file for encrypt')
    encrypt_parser.add_argument('-private_key', required=True,
                                help='Path to private key')
    encrypt_parser.add_argument('-encrypted_sym_key', required=True,
                                help='Path to encrypted symmetric key')
    encrypt_parser.add_argument('-output_file', required=True,
                                help='Path to save encrypted file')

    # parser for decrypt
    decrypt_parser = subparsers.add_parser('decrypt', help='Decrypt file')
    decrypt_parser.add_argument('-input_file', required=True,
                                help='Path to encrypted file')
    decrypt_parser.add_argument('-private_key', required=True,
                                help='Path to private key')
    decrypt_parser.add_argument('-encrypted_sym_key', required=True,
                                help='Path to encrypted symmetric key')
    decrypt_parser.add_argument('-output_file', required=True,
                                help='Path to save decrypted file')

    return parser


def main():
    parser = parsing()
    args = parser.parse_args()

    try:
        if args.command == 'generate':
            CryptoSystem.generate_keys(
                args.key_length,
                args.encrypted_sym_key,
                args.public_key,
                args.private_key
            )

        elif args.command == 'encrypt':
            CryptoSystem.encrypt_file(
                args.input_file,
                args.output_file,
                args.private_key,
                args.encrypted_sym_key,
                )

        elif args.command == 'decrypt':
            CryptoSystem.decrypt_file(
                args.input_file,
                args.output_file,
                args.private_key,
                args.encrypted_sym_key
                )
        else:
            parser.print_help()
    except Exception as e:
        raise Exception(f"Error: {str(e)}")


if __name__ == "__main__":
    main()




# key generator launch template
# python main.py generate -encrypted_sym_key {enc_sym_key.txt} -public_key {public.pem} -private_key {private.pem} -key_length 256
# copy if you want to start key generator part
# python main.py generate -encrypted_sym_key enc_sym_key.txt -public_key public.pem -private_key private.pem -key_length 256

# encrypt launch template
# python main.py encrypt -input_file {input.txt} -private_key {private.pem} -encrypted_sym_key {enc_sym_key.txt} -output_file {encrypted.txt}
# copy if you want to start encrypt part
# python main.py encrypt -input_file input.txt -private_key private.pem -encrypted_sym_key enc_sym_key.txt -output_file encrypted.txt

# decrypt launch template
# python main.py decrypt -input_file {encrypted.txt} -private_key {private.pem} -encrypted_sym_key {enc_sym_key.txt} -output_file {output.txt}
# copy if you want to start decrypt part
# python main.py decrypt -input_file encrypted.txt -private_key private.pem -encrypted_sym_key enc_sym_key.txt -output_file output.txt