import requests
import json


class LabState:
    URL = "https://backend-sapdotten.amvera.io"

    @classmethod
    def get_state(cls) -> bool:
        return json.loads(requests.get(cls.URL + "/state").text)["state"]

    @classmethod
    def set_state(cls, new_state: bool) -> bool:
        return json.loads(
            requests.post(cls.URL + "/change_state", json={"state": new_state}).text
        )["state"]