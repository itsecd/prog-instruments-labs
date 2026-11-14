import re
import csv
import argparse
from collections import Counter

SERVICE_LOGINS = {
    "root", "admin", "administrator", "test", "guest", "user",
    "oracle", "postgres", "mysql", "pi", "ubuntu", "deploy",
    "git", "www-data", "support", "ftpuser", "nagios", "demo",
    "info", "operator",
}

PREFIX_RE = re.compile(
    r"^(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"(?P<day>0?[1-9]|[12]\d|3[01])\s+"
    r"(?P<time>(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d)\s+"
    r"(?P<host>[A-Za-z0-9][A-Za-z0-9._-]*)\s+"
    r"(?P<process>[a-z][a-z0-9_-]*)\[(?P<pid>\d+)\]:\s+"
    r"(?P<message>.*)$"
)
EVENT_RE = re.compile(
    r"^(?P<event>Accepted|Failed|Invalid user|Connection closed by|"
    r"Disconnected from|Received disconnect from|"
    r"error: maximum authentication attempts exceeded for)\b"
)
USER_RE = re.compile(
    r"(?:for\s+invalid\s+user|for|by\s+authenticating\s+user)\s+"
    r"(?P<user>[A-Za-z_][A-Za-z0-9_.-]{0,31})(?=\s|$)"
)
IP_RE = re.compile(
    r"\b(?P<ip>(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})\b"
)
PORT_RE = re.compile(r"\bport\s+(?P<port>\d{1,5})\b")
METHOD_RE = re.compile(r"\b(?P<method>password|publickey|keyboard-interactive)\b")
FINGERPRINT_RE = re.compile(r"\b(?:RSA|ED25519|ECDSA)\s+SHA256:(?P<fp>[A-Za-z0-9+/]{43})")

VALID_EVENTS = {
    "Accepted": "accepted",
    "Failed": "failed",
    "Invalid user": "invalid_user",
    "Connection closed by": "conn_closed",
    "Disconnected from": "disconnected",
    "Received disconnect from": "received_disconnect",
    "error: maximum authentication attempts exceeded for": "max_attempts",
}


def parse_logs(input_file, output_csv, threshold=5):
    events = Counter()
    failed_by_ip = Counter()
    invalid_users = Counter()
    invalid_users_by_ip = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = PREFIX_RE.match(line)
            if not m:
                continue
            msg = m.group('message')
            ev = EVENT_RE.match(msg)
            if not ev:
                continue
            event = VALID_EVENTS[ev.group('event')]
            u = USER_RE.search(msg)
            i = IP_RE.search(msg)
            p = PORT_RE.search(msg)
            meth = METHOD_RE.search(msg)
            fp = FINGERPRINT_RE.search(msg)
            if not u or not i or not p or not (1 <= int(p.group('port')) <= 65535):
                continue
            if event == 'accepted' and not meth:
                continue
            if meth and meth.group('method') == 'publickey' and not fp:
                continue
            user, ip_addr = u.group('user'), i.group('ip')
            events[event] += 1
            if event in ('failed', 'invalid_user'):
                failed_by_ip[ip_addr] += 1
            if event == 'invalid_user':
                invalid_users[user] += 1
                invalid_users_by_ip.setdefault(ip_addr, set()).add(user)

    with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['analysis'])
        for login, cnt in invalid_users.most_common():
            if login.lower() in SERVICE_LOGINS:
                writer.writerow([f"{login}: {cnt} attempts (service login)"])
        for ip, users in invalid_users_by_ip.items():
            if len(users) >= threshold:
                writer.writerow([f"{ip}: {len(users)} logins ({', '.join(sorted(users))})"])
        for ip, cnt in failed_by_ip.most_common():
            if cnt >= threshold:
                writer.writerow([f"{ip}: {cnt} failed attempts"])


def main():
    parser = argparse.ArgumentParser(
        description="SSH auth.log parser. Writes a summary of suspicious activity to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file", help="Path to the source log file")
    parser.add_argument("-o", "--output", default="ssh_events.csv",
                        help="Path to save the CSV summary file")
    parser.add_argument("-t", "--threshold", type=int, default=5,
                        help="Threshold for detecting login enumeration / brute-force")
    args = parser.parse_args()
    parse_logs(args.input_file, args.output, args.threshold)


if __name__ == "__main__":
    main()