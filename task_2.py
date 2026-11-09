import argparse

def take_key(filenamekey):
    """
    Читает файл к ключом
    
    :param filenamekey(str): Имя файла с ключом

    Возвращает:
    dict:Ключ 
    """
    key={}
    with open(filenamekey,"r",encoding="utf-8") as file:
        for line in file:
            line=line.strip()
            if not line:
                continue
            parts=line.split(":",1)
            if len(parts) != 2:
                continue
            key_part=parts[0].strip().strip('"').strip("'")
            value_part=parts[1].strip().strip('"').strip("'")
            key[key_part]=value_part
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
    
def write_result_text(encode_text,filenameoutput):
    """
    Записывает расшифрованный текст в файл
    
    :param encode_text: Расшифрованный текст
    :param filenameoutput: Название файла
    """
    with open(filenameoutput,"w",encoding="utf-8") as file:
        file.write(encode_text)

def write_result_frequency(frequency,filenameoutput):
    """
    Записывает частоты в файл
    
    :param frequency(dict): Словарь частот
    :param filenameoutput(str): Название файла
    """
    with open(filenameoutput,"w",encoding="utf-8") as file:
        for char,value in frequency.items():
            file.write(f"{char}:{value}\n")

def parsing():
    """
    Получение аргументов командной строки
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("filenametext",type=str,help="Введите названия файла с исходным текстом")
    parser.add_argument("filenamekey",type=str,help="Введите название файла с ключом")
    parser.add_argument("filenametextoutput",type=str,help="Введите названия файла с результатом")
    parser.add_argument("filenametextfrequency",type=str,help="Введите названия файла с частотой")
    args=parser.parse_args()
    return args.filenametext,args.filenamekey,args.filenametextoutput,args.filenametextfrequency

def count_frequency_manual(text,key):
    """
    Подсчёт количества появления символов в тексте
    
    :param text: Текст
    :param key: Ключ

    Возвращает:
    dict:Словарь количества появления символов в тексте
    """
    frequency={}
    for char in key.keys():
        frequency[char]=0
    for char in text:
        if char in frequency:
            frequency[char]+=1
    for char,value in frequency.items():
        frequency[char]=value/len(text)
    sorted_frequency = dict(sorted(frequency.items(), key=lambda x: x[1], reverse=True))
    return sorted_frequency

def main():
    filenametext,filenamekey,filenametextoutput,filenametextfrequency=parsing()
    text=take_text(filenametext)
    key=take_key(filenamekey)
    frequency = count_frequency_manual(text,key)
    
    normal_text = ""

    for char in text:
        if char in key:
            normal_text += key[char]
        else:
            normal_text += char
    write_result_text(normal_text,filenametextoutput)
    write_result_frequency(frequency,filenametextfrequency)

if __name__=="__main__":
    main()