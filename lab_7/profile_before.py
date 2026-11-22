import time
import os
import tracemalloc
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
    
    tracemalloc.start()
    
    start_time = time.time()
    encrypt_file_before()
    end_time = time.time()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Время выполнения: {end_time - start_time:.2f} сек")
    print(f"Пиковое потребление RAM: {peak / 10**6:.2f} МБ")