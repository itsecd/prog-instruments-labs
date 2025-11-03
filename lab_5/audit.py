import datetime
import inspect
import json
import os
import socket


def audit_log(event_code, user_id=0, key_id=0, extra_data=None):
    """
    Audit logging.
    :param event_code: code of ivent in UPPERCASE
    :param user_id: user id
    :param key_id: key id
    :param extra_data: extra data
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    service_name = "image_processor"
    hostname = socket.gethostname()

    if extra_data is None:
        extra_data = {}
    
    try:
        frame = inspect.currentframe().f_back
        caller_info = {
            "caller_function": frame.f_code.co_name,
            "caller_filename": os.path.basename(frame.f_code.co_filename),
            "caller_line_number": frame.f_lineno
        }
        extra_data["caller_info"] = caller_info
    except:
        pass

    data_json = json.dumps(extra_data, ensure_ascii=False) if extra_data else "{}"
    
    log_line = f"{timestamp} | {event_code} | {service_name} | {hostname} | {user_id} | {key_id} | {data_json}"
    
    with open("audit.log", "a", encoding="utf-8") as f:
        f.write(log_line + "\n")