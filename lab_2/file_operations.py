import pickle


class FileManager:
    @staticmethod
    def read_file(path):
        """
        Reads and returns the contents of a file

        :param path: Path to the file to read
        :return: Contents of the file
        """
        try:
            with open(path, 'rb') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Error at reading file: {str(e)}")

    @staticmethod
    def write_file(path, data):
        """
        Writes data to a file
        :param path: Path to the file to write
        :param data: Data to write
        """
        try:
            with open(path, 'wb') as f:
                f.write(data)
        except Exception as e:
            raise Exception(f"Error at writing file: {str(e)}")

    @staticmethod
    def save_key(key, path):
        """
        Serializes and saves a cryptographic key to file using pickle

        :param key: Key object to serialize
        :param path: Path to save the key
        """
        try:
            with open(path, 'wb') as f:
                pickle.dump(key, f)
        except Exception as e:
            raise Exception(f"Error at saving key: {str(e)}")

    @staticmethod
    def load_key(path):
        """
        Loads and deserializes a cryptographic key from file

        :param path: Path to the key file
        :return: Deserialized key object
        """
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise Exception(f"Error at loading key: {str(e)}")
