"""
Unit tests for the main functionality of the encryption application.
"""

import argparse
import pytest
from unittest.mock import patch, MagicMock
from main import parse_arguments, create_hybrid_encryption, run_mode, main

def test_parse_arguments_generation_mode() -> None:
    """
    Test the argument parsing for generation mode.
    """
    with patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(
        generation=True, encryption=False, decryption=False,
        key_length=256, text_file="text.txt",
        public_key="public.pem", private_key="private.pem",
        symmetric_key_file="symmetric.txt",
        encrypted_text_file="encrypted.txt",
        decrypted_text_file="decrypted.txt",
    )):
        args = parse_arguments()
        assert args.generation is True
        assert args.encryption is False
        assert args.key_length == 256


@patch("main.HybridEncryption")
@patch("main.SymmetricCryptography")
@patch("main.AsymmetricCryptography")
def test_create_hybrid_encryption(mock_asymmetric_class: MagicMock, mock_symmetric_class: MagicMock, mock_hybrid_class: MagicMock) -> None:
    """
    Test the creation of the hybrid encryption system.
    """
    args = argparse.Namespace(
        text_file="text.txt",
        symmetric_key_file="symmetric.txt",
        encrypted_text_file="encrypted.txt",
        decrypted_text_file="decrypted.txt",
        key_length=256,
        private_key="private.pem",
        public_key="public.pem",
    )

    mock_symmetric_instance = mock_symmetric_class.return_value
    mock_asymmetric_instance = mock_asymmetric_class.return_value

    hybrid_encryption = create_hybrid_encryption(args)

    mock_symmetric_class.assert_called_once_with(args.key_length)
    mock_asymmetric_class.assert_called_once_with(args.private_key, args.public_key)
    mock_hybrid_class.assert_called_once_with(
        args.text_file,
        args.symmetric_key_file,
        args.encrypted_text_file,
        args.decrypted_text_file,
        mock_symmetric_instance,
        mock_asymmetric_instance,
    )

    assert hybrid_encryption == mock_hybrid_class.return_value


@patch("HybridEncryption.HybridEncryption.generate_keys")
@patch("HybridEncryption.HybridEncryption.encrypt_text")
@patch("HybridEncryption.HybridEncryption.decrypt_text")
def test_run_mode(mock_decrypt: MagicMock, mock_encrypt: MagicMock, mock_generate: MagicMock) -> None:
    """
    Test the run mode functionality for all supported modes.
    """
    args = argparse.Namespace(
        generation=True, encryption=False, decryption=False,
        key_length=256, text_file="text.txt",
        public_key="public.pem", private_key="private.pem",
        symmetric_key_file="symmetric.txt",
        encrypted_text_file="encrypted.txt",
        decrypted_text_file="decrypted.txt",
    )
    hybrid_encryption = create_hybrid_encryption(args)

    run_mode(hybrid_encryption, args)
    mock_generate.assert_called_once()
    mock_encrypt.assert_not_called()
    mock_decrypt.assert_not_called()

    args.generation = False
    args.encryption = True
    run_mode(hybrid_encryption, args)
    mock_encrypt.assert_called_once()
    mock_decrypt.assert_not_called()

    args.encryption = False
    args.decryption = True
    run_mode(hybrid_encryption, args)
    mock_encrypt.assert_called_once()
    mock_decrypt.assert_called_once()


@patch('main.parse_arguments')
@patch('main.create_hybrid_encryption')
@patch('main.run_mode')
def test_main(mock_run_mode: MagicMock, mock_create_hybrid_encryption: MagicMock, mock_parse_arguments: MagicMock) -> None:
    """
    Test the main function for correct orchestration of other components.
    """
    mock_parse_arguments.return_value = argparse.Namespace(
        generation=True, encryption=False, decryption=False,
        key_length=256, text_file="text.txt",
        public_key="public.pem", private_key="private.pem",
        symmetric_key_file="symmetric.txt",
        encrypted_text_file="encrypted.txt",
        decrypted_text_file="decrypted.txt",
    )

    main()

    mock_parse_arguments.assert_called_once()
    mock_create_hybrid_encryption.assert_called_once_with(mock_parse_arguments.return_value)
    mock_run_mode.assert_called_once_with(mock_create_hybrid_encryption.return_value, mock_parse_arguments.return_value)
