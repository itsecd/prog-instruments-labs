# -*- coding: utf-8 -*-

class TennisGameDefactored1:
    
    """Класс для управления счетом в теннисной игре для двух игроков.

    Атрибуты:
        SCORE_TO_TEXT (dict): Словарь, сопоставляющий числовые очки их текстовому представлению.
        player1Name (str): Имя первого игрока.
        player2Name (str): Имя второго игрока.
        p1points (int): Текущий счет первого игрока.
        p2points (int): Текущий счет второго игрока.
    """

    SCORE_TO_TEXT = {
    0: "Love",
    1: "Fifteen",
    2: "Thirty",
    3: "Forty"
}

    def _get_score_text(self, score):

        """Преобразует числовой счет в текстовое представление.

        Args:
            score (int): Числовой счет (0-3 или выше).

        Returns:
            str: Текстовое представление счета (например, 'Love', 'Fifteen' или 'Deuce').
        """
        
        return self.SCORE_TO_TEXT.get(score, "Deuce")

    def __init__(self, player1Name, player2Name):
        
        """Инициализирует новую теннисную игру с двумя игроками.

        Args:
            player1Name (str): Имя первого игрока.
            player2Name (str): Имя второго игрока.
        """
        
        self.player1Name = player1Name
        self.player2Name = player2Name
        self.p1points = 0
        self.p2points = 0
        
    def won_point(self, playerName):

        """Добавляет очко игроку, выигравшему розыгрыш.

        Args:
            playerName (str): Имя игрока, который выиграл очко.
        """
        
        if playerName == self.player1Name:
            self.p1points += 1
        else:
            self.p2points += 1
    
    def _format_equal_score(self):

        """Форматирует счет, когда у игроков равное количество очков.

        Returns:
            str: Текстовое представление счета (например, 'Love-All' или 'Deuce').
        """
        
        if self.p1points >= 3:
            return "Deuce"
        return f"{self._get_score_text(self.p1points)}-All"

    def _format_advantage_or_win(self):

        """Форматирует счет для случаев преимущества или победы.

        Returns:
            str: Текстовое представление счета (например, 'Advantage player' или 'Win for player').
        """
        
        score_diff = self.p1points - self.p2points
        if abs(score_diff) >= 2:
            winner = self.player1Name if score_diff > 0 else self.p2points
            return f"Win for {winner}"
        leader = self.player1Name if score_diff > 0 else self.p2points
        return f"Advantage {leader}"

    def _format_regular_score(self):

        """Форматирует обычный счет, когда очки не равны и нет преимущества или победы.

        Returns:
            str: Текстовое представление счета (например, 'Fifteen-Thirty').
        """
        
        p1_text = self._get_score_text(self.p1points)
        p2_text = self._get_score_text(self.p2points)
        return f"{p1_text}-{p2_text}"

    def score(self):

        """Возвращает текущий счет игры в текстовом формате.

        Returns:
            str: Текстовое представление текущего счета.
        """
        
        if self.p1points == self.p2points:
            return self._format_equal_score()
        if self.p1points >= 4 or self.p2points >= 4:
            return self._format_advantage_or_win()
        return self._format_regular_score()


class TennisGameDefactored2:

    """Класс для управления счетом в теннисной игре для двух игроков.

    Атрибуты:
        SCORE_TO_TEXT (dict): Словарь, сопоставляющий числовые очки их текстовому представлению.
        player1Name (str): Имя первого игрока.
        player2Name (str): Имя второго игрока.
        p1points (int): Текущий счет первого игрока.
        p2points (int): Текущий счет второго игрока.
    """
    
    SCORE_TO_TEXT = {
        0: "Love",
        1: "Fifteen",
        2: "Thirty",
        3: "Forty"
    }

    def __init__(self, player1Name, player2Name):

        """Инициализирует новую теннисную игру с двумя игроками.

        Args:
            player1Name (str): Имя первого игрока.
            player2Name (str): Имя второго игрока.
        """
        
        self.player1Name = player1Name
        self.player2Name = player2Name
        self.p1points = 0
        self.p2points = 0
        
    def award_point(self, player_name):

        """Добавляет очко игроку, выигравшему розыгрыш.

        Args:
            player_name (str): Имя игрока, который выиграл очко.
        """
        
        if player_name == self.player1Name:
            self.p1points += 1
        else:
            self.p2points += 1
    
    def _get_score_text(self, score):

        """Преобразует числовой счет в текстовое представление.

        Args:
            score (int): Числовой счет (0-3 или выше).

        Returns:
            str: Текстовое представление счета (например, 'Love', 'Fifteen' или 'Deuce').
        """
        
        return self.SCORE_TO_TEXT.get(score, "Deuce")

    def _format_equal_score(self):

        """Форматирует счет, когда у игроков равное количество очков.

        Returns:
            str: Текстовое представление счета (например, 'Love-All' или 'Deuce').
        """
        
        if self.p1points >= 3:
            return "Deuce"
        return f"{self._get_score_text(self.p1points)}-All"

    def _format_advantage_or_win(self):

        """Форматирует счет для случаев преимущества или победы.

        Returns:
            str: Текстовое представление счета (например, 'Advantage player' или 'Win for player').
        """
        
        score_diff = self.p1points - self.p2points
        if abs(score_diff) >= 2:
            winner = self.player1Name if score_diff > 0 else self.player2Name
            return f"Win for {winner}"
        leader = self.player1Name if score_diff > 0 else self.player2Name
        return f"Advantage {leader}"

    def _format_regular_score(self):

        """Форматирует обычный счет, когда очки не равны и нет преимущества или победы.

        Returns:
            str: Текстовое представление счета (например, 'Fifteen-Thirty').
        """
        
        p1_text = self._get_score_text(self.p1points)
        p2_text = self._get_score_text(self.p2points)
        return f"{p1_text}-{p2_text}"

    def score(self):

        """Возвращает текущий счет игры в текстовом формате.

        Returns:
            str: Текстовое представление текущего счета.
        """
        
        if self.p1points == self.p2points:
            return self._format_equal_score()
        if self.p1points >= 4 or self.p2points >= 4:
            return self._format_advantage_or_win()
        return self._format_regular_score()
    
    def P1Score(self):
        """Добавляет одно очко первому игроку."""
        self.p1points +=1
    
    
    def P2Score(self):
        """Добавляет одно очко второму игроку."""
        self.p2points +=1
        
class TennisGameDefactored3:

    """Класс для управления счетом в теннисной игре для двух игроков.

    Атрибуты:
        SCORE_TO_TEXT (dict): Словарь, сопоставляющий числовые очки их текстовому представлению.
        player1Name (str): Имя первого игрока.
        player2Name (str): Имя второго игрока.
        p1points (int): Текущий счет первого игрока.
        p2points (int): Текущий счет второго игрока.
    """

    SCORE_TO_TEXT = {
        0: "Love",
        1: "Fifteen",
        2: "Thirty",
        3: "Forty"
    }

    def __init__(self, player1Name, player2Name):

        """Инициализирует новую теннисную игру с двумя игроками.

        Args:
            player1Name (str): Имя первого игрока.
            player2Name (str): Имя второго игрока.
        """
        
        self.player1Name = player1Name
        self.player2Name = player2Name
        self.p1points = 0
        self.p2points = 0
        
    def award_point(self, player_name):

        """Добавляет очко игроку, выигравшему розыгрыш.

        Args:
            player_name (str): Имя игрока, который выиграл очко.
        """
        
        if player_name == self.player1Name:
            self.p1points += 1
        else:
            self.p2points += 1
    
    def _get_score_text(self, score):

        """Преобразует числовой счет в текстовое представление.

        Args:
            score (int): Числовой счет (0-3 или выше).

        Returns:
            str: Текстовое представление счета (например, 'Love', 'Fifteen' или 'Deuce').
        """
        
        return self.SCORE_TO_TEXT.get(score, "Deuce")
    
    def _format_equal_score(self):

        """Форматирует счет, когда у игроков равное количество очков.

        Returns:
            str: Текстовое представление счета (например, 'Love-All' или 'Deuce').
        """
        
        if self.p1points >= 3:
            return "Deuce"
        return f"{self._get_score_text(self.p1points)}-All"

    def _format_advantage_or_win(self):

        """Форматирует счет для случаев преимущества или победы.

        Returns:
            str: Текстовое представление счета (например, 'Advantage player' или 'Win for player').
        """
        
        score_diff = self.p1points - self.p2points
        if abs(score_diff) >= 2:
            winner = self.player1Name if score_diff > 0 else self.player2Name
            return f"Win for {winner}"
        leader = self.player1Name if score_diff > 0 else self.player2Name
        return f"Advantage {leader}"

    def _format_regular_score(self):

        """Форматирует обычный счет, когда очки не равны и нет преимущества или победы.

        Returns:
            str: Текстовое представление счета (например, 'Fifteen-Thirty').
        """
        
        p1_text = self._get_score_text(self.p1points)
        p2_text = self._get_score_text(self.p2points)
        return f"{p1_text}-{p2_text}"

    def score(self):

        """Возвращает текущий счет игры в текстовом формате.

        Returns:
            str: Текстовое представление текущего счета.
        """
        
        if self.p1points == self.p2points:
            return self._format_equal_score()
        if self.p1points >= 4 or self.p2points >= 4:
            return self._format_advantage_or_win()
        return self._format_regular_score()

# NOTE: You must change this to point at the one of the three examples that you're working on!
TennisGame = TennisGameDefactored1
