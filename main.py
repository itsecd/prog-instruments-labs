import argparse

import asymmetric_keys_interaction
import symmetric_text_interaction
import data_interaction
import key_generators

def parsing()-> argparse.Namespace:
    """Получение аргументов командной строки"""
    parser=argparse.ArgumentParser()
    
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-gen','--generator',action='store_true',help="Режим генерации ключей")
    group.add_argument('-enc','--encryption',action='store_true',help="Режим шифрования данных")
    group.add_argument('-dec','--decryption',action='store_true',help="Режим дешифрования данных")

    parser.add_argument("--enc_key", type=str,help="Путь к зашифрованному симметричному ключу")
    parser.add_argument("--rsa_pri_key", type=str,help="Путь к закрытому ассимметричному ключу")
    parser.add_argument("--rsa_pub_key", type=str,help="Путь к открытому ассимметричному ключу")
    parser.add_argument("--enc_text", type=str,help="Путь для сохранения(доступа) к зашифрованному тексту")
    parser.add_argument("--dec_text", type=str,help="Путь к расшифрованному тексту")
    parser.add_argument("--init_text", type=str,help="Путь к шифруемому текстовому файлу")
    parser.add_argument("--len_key", type=str,help="Длина ключа")
    parser.add_argument("--config", type=str, default="config.json",help="Путь к файлу настроек(по умолчанию config.json)")

    return parser.parse_args()

def set_settings()->dict[str,str]:
    """Установка конфигурации проекта"""
    args=parsing()
    settings_file_data=data_interaction.load_json(args.config)

    settings= {
        'mode': "gen" if args.generator else "enc" if args.encryption else "dec",
        'initial_file': args.init_text or settings_file_data.get("initial_file"),
        'encrypted_file':args.enc_text or settings_file_data.get("encrypted_file"),
        'decrypted_file':args.dec_text or settings_file_data.get("decrypted_file"),
        'symmetric_key':args.enc_key or settings_file_data.get("symmetric_key"),
        'public_key':args.rsa_pub_key or settings_file_data.get("public_key"),
        'secret_key':args.rsa_pri_key or settings_file_data.get("secret_key"),
        'len_key':args.len_key or settings_file_data.get("len_key"),
    }
    return settings

def main()->None:
    config=set_settings()

    match config:
        case _ if config['mode']=="gen":
            print("Выполение сценария генерации ключей...")
            private_key,public_key=key_generators.generate_asy_key()
            sym_key=key_generators.generate_sym_key(int(config['len_key']))
            data_interaction.save_asy_key(config['public_key'],config['secret_key'],private_key,public_key)
            enc_sym_key=asymmetric_keys_interaction.enc_sym_key(public_key,sym_key)
            data_interaction.save_sym_key(config['symmetric_key'],enc_sym_key)
            print("Выполение сценария закончено...")
        case _ if config['mode']=="enc":
            print("Выполение сценария шифрования данных...")
            dec_sym_key=asymmetric_keys_interaction.dec_sym_key(data_interaction.load_asy_pri_key(config['secret_key']),
                                                                data_interaction.load_sym_key(config['symmetric_key']))
            enc_data=symmetric_text_interaction.encode_text(dec_sym_key,data_interaction.read_text_file(config['initial_file']))
            data_interaction.write_file(enc_data,config['encrypted_file'])
            print("Выполение сценария закончено...")
        case _ if config['mode']=="dec":
            print("Выполение сценария дешифрования данных...")
            dec_sym_key=asymmetric_keys_interaction.dec_sym_key(data_interaction.load_asy_pri_key(config['secret_key']),
                                                                data_interaction.load_sym_key(config['symmetric_key']))
            enc_data=symmetric_text_interaction.decode_text(dec_sym_key,data_interaction.read_file(config['encrypted_file']))
            data_interaction.write_text_file(enc_data,config['decrypted_file'])
            print("Выполение сценария закончено...")

if __name__=="__main__":
    main()