from moduls.asymetric import Asymetric
from moduls.symetric import Symetric
from moduls.reader_writer import Texting
from moduls.logger import *


class cryptography:

    def generation_proc(private_way: str, public_way: str, symm_way: str):
        logger = create_logger()
        logger.info("Запуск процедкры генерации и сериализации ключей")
        sym_key = Symetric.create_sym_key()
        logger.info("Произошло создание симметричного ключа")
        asym_key = Asymetric.create_asym_key()
        logger.info("Произогло создание асимметричного ключа")
        cyph_sym_key = Asymetric.encrypt_sym_key(asym_key, sym_key)
        logger.info("Произошла шифровка симметричного ключа")
        Texting.serialize_private(asym_key, private_way)
        logger.info("Произошла сериализация приватного ключа")
        Texting.serialize_public(asym_key, public_way)
        logger.info("Произошла сериализация пкбличчного ключа")
        Symetric.serialize_sym(cyph_sym_key, symm_way)
        logger.info("Произошла сериализация симметричного ключа")

    def encryption_proc(
        encr_way: str, orig_way: str, private_way: str, public_way: str, symm_way: str
    ):
        logger = create_logger()
        logger.info("Запуск процедуры шифровки текста")
        asym_key = Asymetric.deserylie_asym(public_way, private_way)
        logger.info("Произошла десериализация асимметричного ключа")
        sym_key = Symetric.deserialize_sym(symm_way)
        logger.info("Произошла десериализация симметричного ключа")
        sym_key = Asymetric.decrypt_sym_key(asym_key, sym_key)
        logger.info("Произошло дешифрование симметричного ключа")
        text = Texting.read_file(orig_way)
        logger.info("Произошло чтение оригинального текста")
        enc_text = Symetric.encrypt_text(text, sym_key)
        logger.info("Произошло шифрование оригинального текста")
        Texting.write_bytes(encr_way, enc_text)
        logger.info("Произошла запись зашифрованного текста")

    def decryption_proc(
        uncyph_way: str, encr_way: str, private_way: str, public_way: str, symm_way: str
    ):
        logger = create_logger()
        logger.info("Запуск процедуры дешифровки текста")
        asym_key = Asymetric.deserylie_asym(public_way, private_way)
        logger.info("Произошла десериализация асимметричного ключа")
        sym_key = Symetric.deserialize_sym(symm_way)
        logger.info("Произошла десериализация симметричного ключа")
        sym_key = Asymetric.decrypt_sym_key(asym_key, sym_key)
        logger.info("Произошло дешифрование симметричного ключа")
        text = Texting.read_bytes(encr_way)
        logger.info("Произошло чтение зашифрованного текста")
        text = Symetric.decode_text(text, sym_key)
        logger.info("Произошла дешифровка текста")
        Texting.write_file(uncyph_way, text)
        logger.info("Произошла запись расшифрованного текста")
