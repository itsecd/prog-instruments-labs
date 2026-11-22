with open("big_data.txt", "w", encoding="utf-8") as f:
    for _ in range(200000):
        f.write("Это тестовые данные для проверки производительности гибридной криптосистемы.\n")
print("Файл big_data.txt успешно создан!")