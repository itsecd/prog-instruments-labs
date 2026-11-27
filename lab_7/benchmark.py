import time
from polybius_slow import encrypt as encrypt_slow, decrypt as decrypt_slow
from polybius_optimized import encrypt as encrypt_fast, decrypt as decrypt_fast

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

text = read_file("input_large.txt")

# Замер "ДО"
print("Замер 'ДО' (медленный код):")
start = time.time()
encrypted_slow = encrypt_slow(text)
decrypted_slow = decrypt_slow(encrypted_slow)
time_slow = time.time() - start
print(f"  Время: {time_slow:.4f} сек")
print(f"  Успех: {decrypted_slow == text.upper()}\n")

# Замер "ПОСЛЕ"
print("Замер 'ПОСЛЕ' (оптимизированный код):")
start = time.time()
encrypted_fast = encrypt_fast(text)
decrypted_fast = decrypt_fast(encrypted_fast)
time_fast = time.time() - start
print(f"  Время: {time_fast:.4f} сек")
print(f"  Успех: {decrypted_fast == text.upper()}\n")

# Итог
print(f"Ускорение: {time_slow / time_fast:.2f}x")
