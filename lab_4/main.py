import json
import os
import random
import time
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', 
    handlers=[
        # Log to console
        logging.StreamHandler(), 
        # Log to file
        logging.FileHandler("app.log") 
    ]
)

logger = logging.getLogger(__name__)


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
    """Class for interacting with a user database (using JSON for this example)."""

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
            logger.info(f"Database loaded from {self.db_filename}")
        else:
            self.db = {}
            logger.info(f"Database file {self.db_filename} not found, creating a new one.")

    def save_db(self):
        """
        Saves the database to the file.
        """
        with open(self.db_filename, 'w') as f:
            json.dump(self.db, f, indent=4)
        logger.info(f"Database saved to {self.db_filename}")

    def add_user(self, user: User):
        """
        Adds a new user to the database.

        :param user: The User object to add.
        """
        self.db[user.user_id] = user.__dict__
        self.save_db()
        logger.info(f"User {user.name} with ID {user.user_id} added to the database.")

    def get_user(self, user_id: int) -> Optional[User]:
        """
        Retrieves a user from the database by ID.

        :param user_id: The ID of the user to retrieve.
        :return: The User object, or None if not found.
        """
        user_data = self.db.get(user_id)
        if user_data:
            logger.info(f"User {user_id} found in database.")
            return User(**user_data)
        logger.warning(f"User with ID {user_id} not found in database.")
        return None

    def delete_user(self, user_id: int):
        """
        Deletes a user from the database.

        :param user_id: The ID of the user to delete.
        """
        if user_id in self.db:
            del self.db[user_id]
            self.save_db()
            logger.info(f"User with ID {user_id} deleted from the database.")
        else:
            logger.warning(f"Attempt to delete non-existent user with ID {user_id}.")


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
            logger.info(f"Successfully fetched data for user ID {user_id} from network.")
            return {"user_id": user_id, "name": f"User{user_id}", "email": f"user{user_id}@gmail.com"}
        else:
            logger.error(f"Failed to fetch data for user ID {user_id} from network.")
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
        try:
            user_data = self.network_service.fetch_user_data(user_id)
            user = User(user_id=user_data['user_id'], name=user_data['name'], email=user_data['email'])
            self.user_db.add_user(user)
        except Exception as e:
            logger.error(f"Error occurred while registering user ID {user_id}: {e}")

    def get_user_info(self, user_id: int) -> Optional[User]:
        """
        Retrieves user information.

        :param user_id: The ID of the user.
        :return: The User object, or None if not found.
        """
        user = self.user_db.get_user(user_id)
        if user:
            logger.info(f"User data retrieved: {user}")
        else:
            logger.warning(f"No user data found for user ID {user_id}.")
        return user

    def delete_user(self, user_id: int):
        """
        Deletes a user.

        :param user_id: The ID of the user to delete.
        """
        self.user_db.delete_user(user_id)


class Application:
    """
    Main application class.
    """

    def __init__(self):
        """
        Initializes an Application object.
        """
        self.db_filename = "user_db.json"
        self.user_db = UserDatabase(self.db_filename)
        self.network_service = NetworkService()
        self.user_service = UserService(self.user_db, self.network_service)

    def run(self):
        """
        Runs the application.
        """
        try:
            logger.info("Application started.")
            # Register users
            self.user_service.register_user(1)
            self.user_service.register_user(2)

            # Get user information
            user = self.user_service.get_user_info(1)
            if user:
                logger.info(f"User found: {user}")

            # Delete user
            self.user_service.delete_user(2)
            logger.info("User with ID 2 deleted.")

            # Attempt to get a non-existent user
            user = self.user_service.get_user_info(9999)
            if not user:
                logger.warning("User with ID 9999 not found.")

        except Exception as e:
            logger.error(f"Error occurred during application run: {e}")
        finally:
            logger.info("Application finished.")


if __name__ == "__main__":
    app = Application()
    app.run()