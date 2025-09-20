from file_work import read_text,write_text
from database.db_work import DataBase
import os


def make_keys(user_id: int) -> None:
    """
    Создает приватный и публичый ключи пользователя по пути:
    /etc/wireguard/user_passwords
    :param user_id: id пользователя в телеграме
    :return: None
    """
    try:
        os.system("echo User Private key")
        os.system(f"wg genkey | tee /etc/wireguard/user_passwords/{str(user_id)}_privatekey | wg pubkey | tee /etc/wireguard/user_passwords/{str(user_id)}_publickey")
    except Exception as exc:
        print(f"ERROR! {exc}")


def add_new_peer_to_server_conf(user_id: int, user_ip: str) -> None:
    """
    Записывает данные нового пользователя в конфиг
    :param user_id: id пользователя в телеграме
    :param user_ip: id пользователя в телеграме
    :return: None
    """
    public_key = ""
    with open(f"/etc/wireguard/user_passwords/{str(user_id)}_publickey", "r") as file:
        public_key = file.read()
    peer_str = f'''
[Peer]
PublicKey = {public_key.strip()}
AllowedIPs = {user_ip}'''
    with open("/etc/wireguard/wg0.conf", "a") as file:
        file.write(peer_str)


def make_new_user_conf(user_id: int, user_address: str) -> None:
    """
    Создает новый конфиг пользователя
    :param user_id: id пользователя в телеграме
    :param user_address: id пользователя в телеграме
    :return:
    """
    private_key = ""
    with open(f"/etc/wireguard/user_passwords/{user_id}_privatekey", "r") as file:
        private_key = file.read()

    user_config_str = f'''[Interface]
PrivateKey = {private_key.strip()}
Address = {user_address}
DNS = 8.8.8.8

[Peer]
PublicKey = foeu0p96/OKo3lbdxw7kuFS+aIsP1g7q1KNDO7+tsD8=
Endpoint = 185.58.115.184:51830
AllowedIPs = 0.0.0.0/0
PersistentKeepalive = 20'''
    with open(f"/etc/wireguard/VPN/tg_bot/client_conf/{user_id}wg.conf", "w") as file:
        file.write(user_config_str)


def make_restart_vpn() -> None:
    """
    Перезагружает VPN
    :return:
    """
    os.system("systemctl restart wg-quick@wg0.service")


def delete_user(user_ip: str, user_id: int):
    """
    USE ONLY WITH free_ip FROM db_work
    :param user_ip:
    :param user_id:
    :return:
    """
    data = read_text("/etc/wireguard/wg0.conf")
    public_key = read_text(f"/etc/wireguard/user_passwords/{str(user_id)}_publickey")
    new_data = data.replace(f'''[Peer]
PublicKey = {public_key.strip()}
AllowedIPs = {user_ip}''','')
    write_text("/etc/wireguard/wg0.conf", new_data)
    os.remove(f"/etc/wireguard/user_passwords/{str(user_id)}_publickey")
    os.remove(f"/etc/wireguard/user_passwords/{str(user_id)}_privatekey")
    os.remove(f"/etc/wireguard/VPN/tg_bot/client_conf/{str(user_id)}wg.conf")

