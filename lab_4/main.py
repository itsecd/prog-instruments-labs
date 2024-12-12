import json
import os
import random
import time
from typing import List, Optional


class User:
    """
    Class representing a user.
    """

    def __init__(self, user_id: int, name: str, email: str):
        """
        Initializes a User object.

        :param user_id: The user's ID.
        :param name: The user's name.
        :param email: The user's email address.
        """
        self.user_id = user_id
        self.name = name
        self.email = email

    def __repr__(self):
        return f"User(id={self.user_id}, name={self.name}, email={self.email})"


class UserDatabase:
    """
    Class for interacting with a user database (using JSON for this example).
    """

    def __init__(self, db_filename: str):
        """
        Initializes a UserDatabase object.

        :param db_filename: The path to the database file.
        """
        self.db_filename = db_filename
        self.load_db()

    def load_db(self):
        """
        Loads the database from the file.
        """
        if os.path.exists(self.db_filename):
            with open(self.db_filename, 'r') as f:
                self.db = json.load(f)
        else:
            self.db = {}

    def save_db(self):
        """
        Saves the database to the file.
        """
        with open(self.db_filename, 'w') as f:
            json.dump(self.db, f, indent=4)

    def add_user(self, user: User):
        """
        Adds a new user to the database.

        :param user: The User object to add.
        """
        self.db[user.user_id] = user.__dict__
        self.save_db()

    def get_user(self, user_id: int) -> Optional[User]:
        """
        Retrieves a user from the database by ID.

        :param user_id: The ID of the user to retrieve.
        :return: The User object, or None if not found.
        """
        user_data = self.db.get(user_id)
        if user_data:
            return User(**user_data)
        return None

    def delete_user(self, user_id: int):
        """
        Deletes a user from the database.

        :param user_id: The ID of the user to delete.
        """
        if user_id in self.db:
            del self.db[user_id]
            self.save_db()


class NetworkService:
    """
    Class for simulating network interactions (using a fake API).
    """

    def fetch_user_data(self, user_id: int) -> dict:
        """
        Simulates a network request to fetch user data.

        :param user_id: The ID of the user.
        :return: A dictionary containing user data.
        :raises ValueError: If the data fetch fails.
        """
        # Simulate network latency
        time.sleep(random.uniform(0.5, 1.5)) 
        if random.random() < 0.9:
            return {"user_id": user_id, "name": f"User{user_id}", "email": f"user{user_id}@gmail.com"}
        else:
            raise ValueError(f"Failed to fetch data for user_id {user_id}")


class UserService:
    """
    Class for managing user-related operations.
    """

    def __init__(self, user_db: UserDatabase, network_service: NetworkService):
        """
        Initializes a UserService object.

        :param user_db: The UserDatabase object.
        :param network_service: The NetworkService object.
        """
        self.user_db = user_db
        self.network_service = network_service

    def register_user(self, user_id: int):
        """
        Registers a user using an external service.

        :param user_id: The ID of the user to register.
        """
        user_data = self.network_service.fetch_user_data(user_id)
        user = User(user_id=user_data['user_id'], name=user_data['name'], email=user_data['email'])
        self.user_db.add_user(user)

    def get_user_info(self, user_id: int) -> Optional[User]:
        """
        Retrieves user information.

        :param user_id: The ID of the user.
        :return: The User object, or None if not found.
        """
        user = self.user_db.get_user(user_id)
        if user:
            return user
        return None

    def delete_user(self, user_id: int):
        """
        Deletes a user.

        :param user_id: The ID of the user to delete.
        """
        self.user_db.delete_user(user_id)


class LogService:
    """
    Class for managing log file operations.
    """

    def __init__(self, log_filename: str):
        """
        Initializes a LogService object.

        :param log_filename: The path to the log file.
        """
        self.log_filename = log_filename

    def log(self, message: str):
        """
        Writes a message to the log file.

        :param message: The message to log.
        """
        with open(self.log_filename, 'a') as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


class Application:
    """
    Main application class.
    """

    def __init__(self):
        """
        Initializes an Application object.
        """
        self.db_filename = "user_db.json"
        self.log_filename = "app.log"
        self.user_db = UserDatabase(self.db_filename)
        self.network_service = NetworkService()
        self.user_service = UserService(self.user_db, self.network_service)
        self.log_service = LogService(self.log_filename)

    def run(self):
        """
        Runs the application.
        """
        try:
            self.log_service.log("Application started.")
            self.user_service.register_user(1)
            self.user_service.register_user(2)

            user = self.user_service.get_user_info(1)
            if user:
                self.log_service.log(f"User found: {user}")

            self.user_service.delete_user(2)
            self.log_service.log("User with ID 2 deleted.")

            user = self.user_service.get_user_info(9999)
            if not user:
                self.log_service.log("User with ID 9999 not found.")

        except Exception as e:
            self.log_service.log(f"Error occurred: {e}")
        finally:
            self.log_service.log("Application finished.")


if __name__ == "__main__":
    app = Application()
    app.run()