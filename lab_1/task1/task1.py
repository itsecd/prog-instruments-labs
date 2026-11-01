class TrithemiusCipher:
    def __init__(self)-> None:
        self.alphabet = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ "
        self.alphabet_list = list(self.alphabet)
        self.alphabet_size = len(self.alphabet_list)
        self.tabula_recta = self.create_tabula_recta()
        self.char_to_index = {char: idx for idx, char in enumerate(self.alphabet_list)}
        
    def create_tabula_recta(self) -> list[list[str]]:
        tabula: list[list[str]] = []
        for shift in range(self.alphabet_size):
            row = self.alphabet_list[shift:] + self.alphabet_list[:shift]
            tabula.append(row)
        return tabula
    
    def encrypt(self, text: str) -> str:
        text = text.upper().replace('Ё', 'Е')
        result: list[str] = []
        
        for i, char in enumerate(text):
            if char in self.char_to_index:
                col = self.char_to_index[char]
                row = i % self.alphabet_size
                encrypted_char = self.tabula_recta[row][col]
                result.append(encrypted_char)
            else:
                result.append(char)
        return ''.join(result)
    
    def decrypt(self, text: str) -> str:
        text = text.upper().replace('Ё', 'Е')
        result: list[str] = []
        
        for i, char in enumerate(text):
            if char in self.char_to_index:
                row = i % self.alphabet_size
                encrypted_col = self.tabula_recta[row].index(char)
                decrypted_char = self.alphabet_list[encrypted_col]
                result.append(decrypted_char)
            else:
                result.append(char)
        return ''.join(result)


def load_text(filename: str ='task1/original.txt') -> str:
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def write_text(text: str, file_name: str) -> None:
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(text)

def main() -> None:
    original_text = load_text()
    processed_text = original_text.upper().replace('Ё', 'Е')
    processed_text = ' '.join(processed_text.split())
    
    cipher = TrithemiusCipher()
    encrypted_text = cipher.encrypt(processed_text)
    
    decrypted_text = cipher.decrypt(encrypted_text)
    
    write_text(encrypted_text, 'task1/encrypted.txt')
    write_text(decrypted_text, 'task1/decrypted.txt')


if __name__ == "__main__":
    main()
