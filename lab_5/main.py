from hybrid import HybridCryptography
from functions import ReadWriteParseFunctions
from constants import PATHS


if __name__ == "__main__":
    paths = ReadWriteParseFunctions.json_reader(PATHS)
    crypto_system = HybridCryptography(
        paths["sym_path"], paths["private_path"], paths["public_path"]
    )
    args = ReadWriteParseFunctions.parse()
    if args.key:
        crypto_system.set_paths(args.sym_pth, args.private_pth, args.public_pth)
    match args:
        case args if args.generation:
            crypto_system.generate_keys(int(args.len / 8))
        case args if args.encryption:
            crypto_system.encrypt(args.pth, paths["encrypted_text"])
        case args if args.decryption:
            crypto_system.decrypt(args.cpt, paths["decrypted_text"])
 