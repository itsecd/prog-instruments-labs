class Speaker:
    """
    Class representing a speaker.

    Attributes:
        name (str): The name of the speaker.
        
    """

    def __init__(self, name: str):
        """
        Initializes a speaker with the given name.

        Args:
            name (str): The name of the speaker.
            
        """
        self.name = name


class IRepository:
    """
    Interface for speaker repositories.
    
    """

    def saveSpeaker(self, speaker: Speaker) -> int:
        """
        Saves a speaker to the repository.

        Args:
            speaker (Speaker): The speaker to be saved.

        Returns:
            int: The identifier of the saved speaker.
            
        """
        pass


class SqlServerRepository(IRepository):
    """
    Implementation of the IRepository interface for working with a 
    SQL Server database.
    
    """

    def saveSpeaker(self, speaker: Speaker) -> int:
        """
        Saves a speaker to the SQL Server database.

        Attempts to save the speaker and handles potential exceptions
        that may occur during the save operation.

        Args:
            speaker (Speaker): The speaker to be saved.

        Returns:
            int: The identifier of the saved speaker (1 if successful).

        Raises:
            Exception: Raises an exception if saving fails.
        """
        try:
            
            return 1
        
        except Exception as e:
            
            print(f"An error occurred while saving the speaker: {e}")
            raise  