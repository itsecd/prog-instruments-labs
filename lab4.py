import argparse

alphabet_upper = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ "

def decryption(text,key):
    """
    Расшифровывает текст, зашифрованный шифром Виженера
    
    :param text(str): Зашифрованный текст
    :param key(str): Ключевое слово

    Возвращает:
    str:Расшифрованный текст
    """
    table=[]
    for i in range(len(alphabet_upper)):
        row=alphabet_upper[i:]+alphabet_upper[:i]
        table.append(row)

    decrypted_text=""
    key_counter=0
    for i in range(len(text)):
        char=text[i]
        char_upper=char.upper()

        if char_upper in alphabet_upper:
            key_char=key[key_counter%len(key)]
            key_index=alphabet_upper.index(key_char.upper())

            encode_char=""
            for j  in range(len(table)):
                if(char_upper==table[j][key_index]):
                    encode_char=table[j][0]
            decrypted_text+=encode_char
            
            key_counter+=1
    return decrypted_text

def encode(text,key):
    """
    Зашифровывает текст шифром Виженера
    
    :param text(str): Текст
    :param key(str): Ключевое слово

    Возвращает:
    str:Зашифрованный текст
    """
    table=[]
    for i in range(len(alphabet_upper)):
        row=alphabet_upper[i:]+alphabet_upper[:i]
        table.append(row)

    encode_text=""
    key_counter = 0
    for i in range(len(text)):
        char=text[i]
        char_upper=char.upper()

        if char_upper in alphabet_upper:
            key_char=key[key_counter%len(key)]
            key_index=alphabet_upper.index(key_char.upper())

            char_index=alphabet_upper.index(char_upper)
            encode_char=table[char_index][key_index]
            encode_text+=encode_char
            
            key_counter+=1
    return encode_text

def take_key(filenamekey):
    """
    Читает файл с ключевым словом
    
    :param filenamekey(str): Имя файла с ключевым словом

    Возвращает:
    str:Ключевое слово
    """
    with open(filenamekey,"r",encoding="utf-8") as file:
        key=file.read().strip()
        return key
    
def take_text(filenametext):
    """
    Читает файл с текстом
    
    :param filenametext(str): Имя файла с текстом

    Возвращает:
    str:Текст
    """
    with open(filenametext,"r",encoding="utf-8") as file:
        text=file.read()
        return text
    
def write_result(encode_text,filenameoutput):
    """
    Записывает расшифрованный текст
    
    :param encode_text: Расшифрованный текст
    :param filenameoutput: Название файла
    """
    with open(filenameoutput,"w",encoding="utf-8") as file:
        file.write(encode_text)

def parsing():
    """
    Получение аргументов командной строки
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("filenametext",type=str,help="Введите названия файла с исходным текстом")
    parser.add_argument("filenamekey",type=str,help="Введите название файла с ключом")
    parser.add_argument("filenametextoutput",type=str,help="Введите названия файла с результатом")
    args=parser.parse_args()
    return args.filenametext,args.filenamekey,args.filenametextoutput

def main():
    filenametext,filenamekey,filenametextoutput=parsing()
    key=take_key(filenamekey)
    text=take_text(filenametext)
    encode_text=encode(text,key)
    decryption(encode_text,key)
    write_result(encode_text,filenametextoutput)

if __name__=="__main__":
    main() 
