from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

import crypto_utils as crypto


@pytest.fixture(scope="module")
def rsa_pair():
    return crypto.generate_rsa_keys()


@pytest.fixture
def key_files(tmp_path, rsa_pair):
    private_key, public_key = rsa_pair
    paths = SimpleNamespace(
        private=tmp_path / "keys" / "private.pem",
        public=tmp_path / "keys" / "public.pem",
        encrypted=tmp_path / "keys" / "aes_key.enc",
    )
    crypto.save_private_key(private_key, paths.private)
    crypto.save_public_key(public_key, paths.public)
    return paths


def test_write_bytes_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "file.bin"

    crypto.write_bytes(path, b"\x00\xffpayload")

    assert path.read_bytes() == b"\x00\xffpayload"


def test_writes_and_reads_relative_file_without_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    crypto.write_bytes("file.bin", b"\x00\xff")

    assert crypto.read_bytes("file.bin") == b"\x00\xff"


@pytest.mark.parametrize(("bits", "byte_count"), [(128, 16), (192, 24), (256, 32)])
def test_aes_key_uses_requested_number_of_random_bytes(monkeypatch, bits, byte_count):
    calls = []

    def random_bytes(size):
        calls.append(size)
        return b"\x07" * size

    monkeypatch.setattr(crypto, "os", SimpleNamespace(urandom=random_bytes))

    assert crypto.generate_symmetric_key(bits) == b"\x07" * byte_count
    assert calls == [byte_count]


@pytest.mark.parametrize("bits", [0, 127, 257, "128"])
def test_rejects_unsupported_aes_key_size(bits):
    with pytest.raises(ValueError, match="128, 192 или 256"):
        crypto.generate_symmetric_key(bits)


def test_rsa_key_pair_has_matching_public_numbers(rsa_pair):
    private_key, public_key = rsa_pair

    assert private_key.key_size == public_key.key_size == 2048
    assert public_key.public_numbers().e == 65537
    assert private_key.public_key().public_numbers() == public_key.public_numbers()


def test_pem_files_preserve_rsa_key_numbers(key_files, rsa_pair):
    private_key, public_key = rsa_pair

    assert crypto.load_private_key(key_files.private).private_numbers() == private_key.private_numbers()
    assert crypto.load_public_key(key_files.public).public_numbers() == public_key.public_numbers()


def test_generates_matching_key_files_and_wrapped_aes_key(tmp_path, monkeypatch, rsa_pair):
    monkeypatch.setattr(crypto, "generate_rsa_keys", lambda: rsa_pair)
    public_path = tmp_path / "public.pem"
    private_path = tmp_path / "private.pem"
    encrypted_path = tmp_path / "aes_key.enc"

    crypto.generate_keys(encrypted_path, public_path, private_path, 192)

    private_key = crypto.load_private_key(private_path)
    public_key = crypto.load_public_key(public_path)
    wrapped_key = encrypted_path.read_bytes()
    assert public_key.public_numbers() == private_key.public_key().public_numbers()
    assert len(wrapped_key) == 256
    assert len(crypto.decrypt_symmetric_key(wrapped_key, private_key)) == 24


@pytest.mark.parametrize(
    ("bits", "payload", "ciphertext_size"),
    [
        pytest.param(128, b"", 32, id="empty"),
        pytest.param(192, bytes(range(16)), 48, id="full-block"),
        pytest.param(256, bytes(range(256)) + b"\xff", 288, id="multiple-blocks"),
    ],
)
def test_file_round_trip_preserves_exact_bytes(tmp_path, key_files, bits, payload, ciphertext_size):
    source = tmp_path / "input.bin"
    encrypted = tmp_path / "ciphertext" / "encrypted.bin"
    restored = tmp_path / "restored" / "output.bin"
    source.write_bytes(payload)

    crypto.encrypt_file(source, key_files.public, key_files.encrypted, encrypted, bits)
    crypto.decrypt_file(encrypted, key_files.private, key_files.encrypted, restored)

    assert restored.read_bytes() == payload
    assert source.read_bytes() == payload
    assert len(encrypted.read_bytes()) == ciphertext_size


def test_decryption_rejects_wrapped_key_for_other_rsa_pair(tmp_path, key_files, rsa_pair):
    _, public_key = rsa_pair
    key_files.encrypted.write_bytes(crypto.encrypt_symmetric_key(b"\x01" * 16, public_key))
    other_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    crypto.save_private_key(other_private_key, key_files.private)
    restored = tmp_path / "output.bin"
    ciphertext = tmp_path / "encrypted.bin"
    ciphertext.write_bytes(b"\x00" * 32)

    with pytest.raises(ValueError):
        crypto.decrypt_file(ciphertext, key_files.private, key_files.encrypted, restored)

    assert not restored.exists()


def test_decryption_rejects_ciphertext_without_full_iv(tmp_path, key_files, rsa_pair):
    _, public_key = rsa_pair
    key_files.encrypted.write_bytes(crypto.encrypt_symmetric_key(b"\x01" * 16, public_key))
    encrypted = tmp_path / "encrypted.bin"
    restored = tmp_path / "output.bin"
    encrypted.write_bytes(b"\x00" * 15)

    with pytest.raises(ValueError):
        crypto.decrypt_file(encrypted, key_files.private, key_files.encrypted, restored)

    assert not restored.exists()
