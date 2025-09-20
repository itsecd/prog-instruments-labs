import sqlite3


class DataBase:
    def __connect_database(self, db_name: str) -> None:
        """
        Подключается к базе данных
        :param db_name:  название базы
        :return:
        """
        try:
            self.database = sqlite3.connect(db_name)
            self.cursor = self.database.cursor()
        except Exception as exc:
            print(f"ERROR! Can not connect to data base {exc}")

    def __init__(self, db_name: str):
        try:
            self.database = sqlite3.connect(db_name)
            self.cursor = self.database.cursor()
            self.database.close()
        except Exception as exc:
            print(f"ERROR! Can not init data base {exc}")

    def make_users_table(self, db_name: str) -> None:
        """
        Создание таблицы пользователей в базе
        :param db_name: название базы
        :return:
        """
        try:
            self.__connect_database(db_name)
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY UNIQUE,
            activ BLOB,
            address TEXT
            )
            ''')
            self.database.commit()
            self.cursor.execute('INSERT INTO users (id, activ, address) VALUES (?, ?, ?)', (0, 1, "10.0.0.1/24"))
            self.database.commit()
            self.database.close()
        except Exception as exc:
            print(f"ERROR! Can not make users table in data base {exc}")

    def insert_new_user(self, user_id: int, db_name: str) -> None:
        """
        Добовление нового пользователя в базу
        :param user_id: id пользователя в телеграме
        :param db_name: название базы
        :return:
        """
        self.__connect_database(db_name)
        try:
            self.cursor.execute('INSERT INTO users (id, activ, address) VALUES (?, ?, ?)', (user_id, 0, ""))
            self.database.commit()
        except Exception as exc:
            print(f"User already exists: {exc}")
        self.database.close()

    def get_not_activ_user(self, db_name) -> list:
        """
        Возвращает массив не активированых пользователей
        :param db_name: название базы
        :return: лист
        """
        try:
            self.__connect_database(db_name)
            self.cursor.execute('SELECT id, activ FROM users WHERE activ = 0')
            not_active_users = self.cursor.fetchall()
            self.database.close()
            return not_active_users
        except Exception as exc:
            print(f"ERROR! Can not take not active users {exc}")
        finally:
            return not_active_users

    def get_used_ip(self, db_name: str) -> list:
        """
        Возвращает массив используемых ip
        :param db_name: название базы
        :return: list[ip: str]
        """
        try:
            self.__connect_database(db_name)
            self.cursor.execute('SELECT address FROM users')
            used_ip = self.cursor.fetchall()
            self.database.close()
            used_ip_list = []
            for ip in used_ip:
                used_ip_list.append(ip[0])
            return used_ip_list
        except Exception as exc:
            print(f"ERROR! Can not take used ip {exc}")
        finally:
            return used_ip_list

    def add_used_ip(self, db_name: str, user_ip: str, user_id: int) -> None:
        """
        Присваивает пользователя ip и указывает что он активирован
        :param db_name: название базы
        :param user_ip: ip пользователя
        :param user_id: id пользователя в телеграме
        :return:
        """
        self.__connect_database(db_name)
        try:
            self.cursor.execute(f"UPDATE users SET address = '{user_ip}' WHERE id = {str(user_id)}")
            self.cursor.execute(f"UPDATE users SET activ = 1 WHERE id = {str(user_id)}")
            self.database.commit()
            self.database.close()
        except Exception as exc:
            print(f"ERROR!Can not add ip address to user {user_id}: {exc}")

    def is_activ(self, db_name: str, user_id: int) -> bool:
        """
        Проверяет активирован ли пользователь, Если активирован то True
        :param db_name: название базы
        :param user_id: id пользователя в телеграме
        :return:
        """
        self.__connect_database(db_name)
        try:
            self.cursor.execute(f"SELECT activ FROM users WHERE id = {user_id}")
            flag = self.cursor.fetchall()
            self.database.close()
            flag2 = flag[0]
            if flag2[0] == 1:
                return True
            return False
        except Exception as exc:
            print(f"ERROR!Can not take info about user id({user_id}: {exc})")


    def get_ip_by_id(self, db_name: str, user_id: int) -> str:
        """
        Возвращает ip по id
        :param db_name: название базы
        :param user_id: id пользователя в телеграме
        :return:
        """
        self.__connect_database(db_name)
        try:
            self.cursor.execute(f"SELECT address FROM users WHERE id = {user_id}")
            ip = self.cursor.fetchall()
            self.database.close()
            clean_ip = ip[0]
            return clean_ip[0]
        except Exception as exc:
            print(f"ERROR! Can not take ip from user id({user_id}: {exc})")

    def free_ip(self, db_name: str, user_id: int) -> None:
        """
        Освобождает IP
        :param db_name: названиме базы
        :param user_id: id пользователя в телеграме
        :return:
        """
        self.__connect_database(db_name)
        try:
            self.cursor.execute(f'UPDATE users SET address = "", activ = 0 WHERE id = {user_id}')
            self.database.commit()
            self.database.close()
        except Exception as exc:
            print(f"ERROR! Can not delit ip address from user id({user_id}: {exc})")

    def is_user_in_db(self, db_name: str, user_id: int) -> bool:
        self.__connect_database(db_name)
        try:
            self.cursor.execute(f'SELECT EXISTS(SELECT 1 FROM users WHERE id = {user_id})')
            flag = self.cursor.fetchall()[0]
            return flag[0]
        except Exception as exc:
            print(f"ERROR! Can not check user({user_id}): {exc}")





