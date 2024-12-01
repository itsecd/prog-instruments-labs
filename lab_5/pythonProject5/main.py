import argparse
import file_operation
import assymetric
import symmetric

ALGORITHMS = ['CAST5', 'RSA']
ENC_OR_DEC = ['encryption', 'decryption']

"""
Модуль для шифрования и дешифрования данных с использованием алгоритмов RSA и CAST5.
Этот модуль содержит функцию main(), которая выполняет шифрование 
или дешифрование данных в соответствии с переданными параметрами.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-alg', '--algorithm', help='Выбор протокола шифрования.', required=True)
    parser.add_argument('-ed', '--enc_or_dec', help='Расшифровка или зашифровка.', required=True)
    parser.add_argument('-kz', '--key_size', help='Размер ключа для шифрования.')
    parser.add_argument('-k', '--key', help='Откуда берётся ключ при зашифровке.')
    parser.add_argument('-s', '--settings', help='Путь к json файлу с путями.', required=True)
    args = parser.parse_args()
    while args.algorithm not in ALGORITHMS:
        args.algorithm = input('Название алгоритма введено неверно. Возможные варианты:RSA или CAST5. Введите ещё раз:')
    while args.enc_or_dec not in ENC_OR_DEC:
        args.enc_or_dec = input('Ошибка. Возможные варианты: encryption или decryption. Попробуйте ввести ещё раз: ')
    if args.key != 'file' and args.key != 'generate' and args.enc_or_dec == 'encryption':
        raise SystemExit

    settings = file_operation.read_json(args.settings)

    match args.enc_or_dec:
        case 'encryption':
            text = file_operation.read_file(settings['text'])
        case 'decryption':
            text = file_operation.read_bytes_from_file(settings['text'])
    match args.algorithm:
        case 'RSA':
            RSA = assymetric.RSA()
            match args.enc_or_dec:
                case 'encryption':
                    match args.key:
                        case 'file':
                            file_operation.get_open_key_from_file(settings['rsa_public_key'])
                        case 'generate':
                            while not args.key_size.isdigit() or int(args.key_size) % 8 != 0 or int(args.key_size) < 2048:
                                args.key_size = input('Ключ RSA введен неверно.\n'
                                                      'Попробуйте ввести ещё раз: ')
                            RSA.generate_key(int(args.key_size))
                            file_operation.get_pub_and_priv_key_to_file(RSA, settings['rsa_public_key'], settings['rsa_private_key'])
                    bytes_ = bytes(text, 'UTF-8')
                    c_text = RSA.encrypt_bytes(bytes_)
                    file_operation.write_bytes_to_file(settings['text_after_encryption_rsa'], c_text)
                case 'decryption':
                    file_operation.get_private_key_from_file(RSA, settings['rsa_private_key'])
                    cc_text = RSA.decrypt_bytes(text).decode('UTF-8')
                    file_operation.write_file(settings['text_after_decryption_rsa'], cc_text)
        case 'CAST5':
            CAST5 = symmetric.CAST_5()
            match args.enc_or_dec:
                case 'encryption':
                    match args.key:
                        case 'file':
                            file_operation.get_key_from_file(settings['CAST5_key'])
                        case 'generate':
                            while not args.key_size.isdigit() or not 5 <= int(args.key_size) <= 16:
                                args.key_size = input('Ключ CAST5 должен быть натуральным '
                                                      'числом, в диапазоне [5, 16].\n'
                                                      'Попробуйте ввести ещё раз: ')
                            CAST5.generate_key()
                            file_operation.get_key_to_file(CAST5, settings['CAST5_key'])
                    bytes_ = bytes(text, 'UTF-8')
                    c_text = CAST5.encrypt_bytes(bytes_)
                    file_operation.write_bytes_to_file(settings['text_after_encryption_cast5'], c_text)
                case 'decryption':
                    file_operation.get_key_from_file(CAST5, settings['CAST5_key'])
                    cc_text = CAST5.decrypt_bytes(text).decode('UTF-8', errors='ignore')
                    file_operation.write_file(settings['text_after_decryption_cast5'], cc_text)


if __name__ == '__main__':
    main()