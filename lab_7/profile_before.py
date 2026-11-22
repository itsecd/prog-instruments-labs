import time
import os
from memory_profiler import memory_usage
from crypto.hybrid import HybridCrypto
import json

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

hybrid = HybridCrypto(config)
private_key, public_key = hybrid.asymmetric.generate_rsa_keypair()

def encrypt_file_before():
    with open("big_data.txt", "rb") as f:
        data = f.read()
    
    encrypted_data, encrypted_key = hybrid.encrypt_hybrid(data, "ChaCha20", public_key)
    return encrypted_data

if __name__ == "__main__":
    print(f"Размер файла: {os.path.getsize('big_data.txt') / (1024*1024):.2f} МБ")
    
    start_time = time.time()
    mem_usage = memory_usage((encrypt_file_before, ()), max_usage=True)
    end_time = time.time()

    print(f"Время выполнения: {end_time - start_time:.2f} сек")
    print(f"Пиковое потребление RAM: {mem_usage:.2f} МБ")