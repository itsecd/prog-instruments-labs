import os

from database.db_work import DataBase
from file_work import read_json
import telebot
from IPGenerator import IPGenerator
from client_scripts.client_scripts import make_keys, make_new_user_conf, add_new_peer_to_server_conf,make_restart_vpn,delete_user

SETTINGS_JSON = read_json("/etc/wireguard/VPN/tg_bot/settings.json")
TOKEN = SETTINGS_JSON["TOKEN"]
bot = telebot.TeleBot(TOKEN)

db = DataBase(SETTINGS_JSON["user_db"])
db.make_users_table(SETTINGS_JSON["user_db"])


ip_generator = IPGenerator()

@bot.message_handler(commands=["start"])
def main(message):
    bot.send_message(message.chat.id, '''Привет, я выдаю конфиги для VPN, скачай WireGuard и вставь туда мой конфиг

Для создания конфига пиши:
/make_conf''')
    print(message.from_user.id)
    db.insert_new_user(message.from_user.id, SETTINGS_JSON["user_db"])

@bot.message_handler(commands=["help"])
def help_func(message):
    bot.send_message(message.chat.id, '''<b>Информация</b>
Для использования VPN, вам нужно скачать приложение WireGuard
Есть в App Store и Google Play
После вставить конфиг из бота в приложение


<b>Обзор команд</b>
/get_conf - Выдает ваш конфиг, если он уже создавался вами
/make_conf - Создает ваш конфиг''', parse_mode="html")


@bot.message_handler(commands=["get_conf"])
def get_conf(message):
    try:
        bot.send_message(message.chat.id, "Секунду")
        with open(f"/etc/wireguard/VPN/tg_bot/client_conf/{str(message.from_user.id)}wg.conf", "rb") as file:
            bot.send_document(message.chat.id, file)
    except Exception as exc:
        print(f"ERROR! in func get_conf: {exc}")
        bot.send_message(message.chat.id, '''Не нашел твой конфиг, сорян
        Ты его создавал?''')


@bot.message_handler(commands=["make_conf"])
def make_conf(message):
    try:
        user_id = message.from_user.id
        if not (db.is_user_in_db(SETTINGS_JSON["user_db"], user_id)):
            db.insert_new_user(user_id, SETTINGS_JSON["user_db"])
        if not db.is_activ(SETTINGS_JSON["user_db"], user_id):
            bot.send_message(message.chat.id, "Создаю ваш конфиг")
            make_keys(user_id)
            new_user_ip = ip_generator.get_new_ip(db, SETTINGS_JSON["user_db"])
            add_new_peer_to_server_conf(user_id, new_user_ip)
            make_new_user_conf(user_id, new_user_ip)
            db.add_used_ip(SETTINGS_JSON["user_db"], new_user_ip, user_id)
            with open(f"/etc/wireguard/VPN/tg_bot/client_conf/{user_id}wg.conf", "rb") as file:
                bot.send_document(message.chat.id, file)
            make_restart_vpn()
        else:
            get_conf(message)
    except Exception as exc:
        print(f"ERROR!Can not make config for user({user_id}): {exc}")


if __name__ == "__main__":
    bot.polling(non_stop=True)
