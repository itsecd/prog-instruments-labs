import requests
import json
import logging


class LabState:
    URL = "https://lababackend-sapdotten.amvera.io"

    @classmethod
    def get_state(cls) -> bool:
        logging.info(f"Попытка get запроса состояния лаборатории к бэкенду.")
        try:
            request = requests.get(cls.URL + "/state")
            state = json.loads(request.text)["state"]
            logging.info(f"Ответ от бэкенда получен: {state}")
            return json.loads(request.text)["state"]
        except requests.exceptions.HTTPError as err:
            logging.error(f"HTTP ошибка: {err}. URL: {cls.URL}")
        except requests.exceptions.ConnectionError as err:
            logging.error(f"Ошибка подключения к {cls.URL}: {err}")
        except requests.exceptions.Timeout as err:
            logging.error(f"Превышено время ожидания запроса для {cls.URL}")
        except requests.exceptions.RequestException as err:
            logging.error(f"Неопознанная ошибка запроса {cls.URL}: {err}")
        except Exception as err:
            logging.critical(f"Неопознанная ошибка: {err}")
        return False

    @classmethod
    def set_state(cls, new_state: bool) -> bool:
        logging.info(f"Попытка post запроса изменения состояния лаборатории к бэкенду.")
        try:
            request = requests.post(cls.URL + "/change_state", json={"state": new_state})
            state = json.loads(request.text)["state"]
            logging.info(f"Ответ от бэкенда получен: {state}")
            return state
        except requests.exceptions.HTTPError as err:
            logging.error(f"HTTP ошибка: {err}. URL: {cls.URL}")
        except requests.exceptions.ConnectionError as err:
            logging.error(f"Ошибка подключения к {cls.URL}: {err}")
        except requests.exceptions.Timeout as err:
            logging.error(f"Превышено время ожидания запроса для {cls.URL}")
        except requests.exceptions.RequestException as err:
            logging.error(f"Неопознанная ошибка запроса{cls.URL}: {err}")
        except Exception as err:
            logging.critical(f"Неопознанная ошибка: {err}")
        return False