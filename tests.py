import pytest
import key_generators
import asymmetric_keys_interaction
import symmetric_text_interaction
import data_interaction
import main

def test_generate_sym_key():
    sym_key=key_generators.generate_sym_key(40)
    assert sym_key is not None
    assert len(sym_key)==5

def test_generate_asy_key():
    private_key,public_key=key_generators.generate_asy_key()
    assert public_key is not None
    assert private_key is not None

def test_asymmetric_keys_interaction():
    private_key,public_key=key_generators.generate_asy_key()
    sym_key=key_generators.generate_sym_key(40)
    enc_key =asymmetric_keys_interaction.enc_sym_key(public_key,sym_key)
    dec_key=asymmetric_keys_interaction.dec_sym_key(private_key,enc_key)
    assert dec_key==sym_key

def test_load_json():
    data=data_interaction.load_json("config.json")
    assert data is not None
    assert data.get("initial_file")=="init.txt"
    assert data.get("symmetric_key")=="sym_key.enc"

def test_save_and_load_sym_key():
    sym_key=key_generators.generate_sym_key(40)
    data_interaction.save_sym_key("sym_key.enc",sym_key)
    assert sym_key==data_interaction.load_sym_key("sym_key.enc")

def test_save_and_load_asm_key():
    private_key,public_key=key_generators.generate_asy_key()
    data_interaction.save_asy_key("pub_key.pem","pri_key.pem",private_key,public_key)
    loaded_pub = data_interaction.load_asy_pub_key("pub_key.pem")
    loaded_pri = data_interaction.load_asy_pri_key("pri_key.pem")
    
    assert loaded_pub.public_numbers() == public_key.public_numbers()
    assert loaded_pri.private_numbers() == private_key.private_numbers()

@pytest.mark.parametrize("len_key, init_text",[
    (32, "Hello, world!"),
    (128, "Тест фываы ловавыфаф !@#$%^&*()_+"),
    (256, ""),
    (448, "Текст " * 50),
])

def test_sym_decode_encode_text(len_key,init_text):
    sym_key=key_generators.generate_sym_key(len_key)
    enc_text = symmetric_text_interaction.encode_text(sym_key,init_text)
    assert enc_text!=init_text.encode("utf-8")
    dec_text = symmetric_text_interaction.decode_text(sym_key,enc_text)
    assert dec_text==init_text

def test_gen_main(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "-enc", "--len_key", "128"])
    monkeypatch.setattr(data_interaction, "load_json", lambda path: {"initial_file": "init.txt", "len_key": 40})

    settings = main.set_settings()

    assert settings["mode"] == "enc"
    assert settings["len_key"] == "128" 
    assert settings["initial_file"] == "init.txt"