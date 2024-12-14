from moduls.asymetric import Asymetric
from moduls.symetric import Symetric
from moduls.reader_writer import Texting


class cryptography:

    def generation_proc(private_way: str, public_way: str, symm_way: str):
        print("Запуск процедкры генерации и сериализации ключей")
        sym_key = Symetric.create_sym_key()
        print("Произошло создание симметричного ключа")
        asym_key = Asymetric.create_asym_key()
        print("Произогло создание асимметричного ключа")
        cyph_sym_key = Asymetric.encrypt_sym_key(asym_key, sym_key)
        print("Произошла шифровка симметричного ключа")
        Texting.serialize_private(asym_key, private_way)
        print("Произошла сериализация приватного ключа")
        Texting.serialize_public(asym_key, public_way)
        print("Произошла сериализация пкбличчного ключа")
        Symetric.serialize_sym(cyph_sym_key, symm_way)
        print("Произошла сериализация симметричного ключа")

    def encryption_proc(
        encr_way: str, orig_way: str, private_way: str, public_way: str, symm_way: str
    ):
        print("Запуск процедуры шифровки текста")
        asym_key = Asymetric.deserylie_asym(public_way, private_way)
        print("Произошла десериализация асимметричного ключа")
        sym_key = Symetric.deserialize_sym(symm_way)
        print("Произошла десериализация симметричного ключа")
        sym_key = Asymetric.decrypt_sym_key(asym_key, sym_key)
        print("Произошло дешифрование симметричного ключа")
        text = Texting.read_file(orig_way)
        print("Произошло чтение оригинального текста")
        enc_text = Symetric.encrypt_text(text, sym_key)
        print("Произошло шифрование оригинального текста")
        Texting.write_bytes(encr_way, enc_text)
        print("Произошла запись зашифрованного текста")

    def decryption_proc(
        uncyph_way: str, encr_way: str, private_way: str, public_way: str, symm_way: str
    ):
        print("Запуск процедуры дешифровки текста")
        asym_key = Asymetric.deserylie_asym(public_way, private_way)
        print("Произошла десериализация асимметричного ключа")
        sym_key = Symetric.deserialize_sym(symm_way)
        print("Произошла десериализация симметричного ключа")
        sym_key = Asymetric.decrypt_sym_key(asym_key, sym_key)
        print("Произошло дешифрование симметричного ключа")
        text = Texting.read_bytes(encr_way)
        print("Произошло чтение зашифрованного текста")
        text = Symetric.decode_text(text, sym_key)
        print("Произошла дешифровка текста")
        Texting.write_file(uncyph_way, text)
        print("Произошла запись расшифрованного текста")
