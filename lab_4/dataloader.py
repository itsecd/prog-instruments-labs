import os
import pandas as pd
from torchtext import data


class PoetryDataProcessor:
    def __init__(self, dataset_path="../dataset/poetry"):
        self.dataset_path = dataset_path
        self.train_csv = None

    def prepare_data(self, debug=False):
        """Подготавливает данные для обучения"""
        if debug:
            return self._create_small_dataset()
        else:
            return self._prepare_full_dataset()

    def _prepare_full_dataset(self):
        """Подготавливает полный набор данных"""
        target_path = os.path.join(self.dataset_path, "train.csv")

        if not os.path.exists(target_path):
            self._convert_txt_to_csv(target_path)

        return target_path

    def _create_small_dataset(self):
        """Создает уменьшенный набор данных для отладки"""
        target_path = os.path.join(self.dataset_path, "train_small.csv")

        if not os.path.exists(target_path):
            full_data_path = self._prepare_full_dataset()
            df = pd.read_csv(full_data_path)
            # Берем только первые 100 примеров для отладки
            df_small = df.head(100)
            df_small.to_csv(target_path, index=False, encoding='utf_8_sig')

        return target_path

    def _convert_txt_to_csv(self, output_path):
        """Конвертирует TXT файл в CSV"""
        file_path = os.path.join(self.dataset_path, "poetryFromTang.txt")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден")

        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        poems = content.split("\n\n")
        cleaned_poems = [poem.replace("\n", "") for poem in poems if poem.strip()]

        df = pd.DataFrame({"sent": cleaned_poems})
        df.to_csv(output_path, index=False, encoding='utf_8_sig')

        print(f"Данные сохранены в {output_path}, всего стихов: {len(cleaned_poems)}")

    def create_dataloader(self, batch_size=32, debug=False):
        """Создает DataLoader для обучения"""
        data_path = self.prepare_data(debug=debug)

        def tokenizer(text):
            return list(text)

        # Определяем поля
        SENT = data.Field(
            sequential=True,
            tokenize=tokenizer,
            lower=False,
            init_token="<start>",
            eos_token="<end>"
        )

        # Создаем dataset
        train_data, valid_data = data.TabularDataset.splits(
            path='',
            train=data_path,
            validation=data_path,
            format='csv',
            skip_header=True,
            fields=[('sent', SENT)]
        )

        # Строим словарь
        SENT.build_vocab(train_data)

        # Создаем DataLoader
        train_loader = data.BucketIterator(
            train_data,
            batch_size=batch_size,
            sort_key=lambda x: len(x.sent),
            shuffle=True
        )

        return train_loader, SENT.vocab


def create_dataloader(batch_size=32, debug=False, dataset_path="../dataset/poetry"):
    """Функция для быстрого создания DataLoader"""
    processor = PoetryDataProcessor(dataset_path)
    return processor.create_dataloader(batch_size=batch_size, debug=debug)


if __name__ == "__main__":
    train_loader, vocabulary = create_dataloader()

    for batch in train_loader:
        print("Размер батча:", batch.sent.t().shape)
        print("Пример данных:", batch.sent.t()[0])
        break
