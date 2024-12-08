import argparse
import os

from HybridEncryption import HybridEncryption
from SymmetricCryptography import SymmetricCryptography
from AsymmetricCryptography import AsymmetricCryptography

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Single entry point for key generation, encryption, and decryption.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-gen', '--generation',
                       action='store_true',
                       help='Run key generation mode.')
    group.add_argument('-enc', '--encryption',
                       action='store_true',
                       help='Run encryption mode.')
    group.add_argument('-dec', '--decryption',
                       action='store_true',
                       help='Run decryption mode.')

    parser.add_argument('-k', '--key_length',
                        type=int,
                        default=256,
                        help='Length of the symmetric key in bits (default: 256).')

    parser.add_argument('-t', '--text_file',
                        type=str,
                        default=os.path.join('text.txt'),
                        help='Path to the text file (default: text.txt).')

    parser.add_argument('-pk', '--public_key',
                        type=str,
                        default=os.path.join('asymmetric_keys', 'public.pem'),
                        help='Path to the public key file (default: asymmetric_keys/public.pem).')

    parser.add_argument('-sk', '--private_key',
                        type=str,
                        default=os.path.join('asymmetric_keys', 'private.pem'),
                        help='Path to the private key file (default: asymmetric_keys/private.pem).')

    parser.add_argument('-skf', '--symmetric_key_file',
                        type=str,
                        default=os.path.join('symmetric_keys', 'symmetric.txt'),
                        help='Path to the symmetric key file (default: symmetric_keys/symmetric.txt).')

    parser.add_argument('-et', '--encrypted_text_file',
                        type=str,
                        default=os.path.join('encrypted_text.txt'),
                        help='Path to the encrypted text file (default: encrypted_text.txt).')

    parser.add_argument('-dt', '--decrypted_text_file',
                        type=str,
                        default=os.path.join('decrypted_text.txt'),
                        help='Path to the decrypted text file (default: decrypted_text.txt).')

    return parser.parse_args()


def create_hybrid_encryption(args):
    """
    Create and return a HybridEncryption instance based on the provided arguments.
    """
    symmetric_crypto = SymmetricCryptography(args.key_length)
    asymmetric_crypto = AsymmetricCryptography(args.private_key, args.public_key)
    return HybridEncryption(
        args.text_file,
        args.symmetric_key_file,
        args.encrypted_text_file,
        args.decrypted_text_file,
        symmetric_crypto,
        asymmetric_crypto,
    )


def run_mode(hybrid_encryption, args):
    """
    Run the specified mode based on command-line arguments.
    """
    if args.generation:
        hybrid_encryption.generate_keys()
    elif args.encryption:
        hybrid_encryption.encrypt_text()
    elif args.decryption:
        hybrid_encryption.decrypt_text()


def main():
    """
    Main entry point for the script.
    """
    args = parse_arguments()
    hybrid_encryption = create_hybrid_encryption(args)
    run_mode(hybrid_encryption, args)