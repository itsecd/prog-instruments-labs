"""Модуль для ведения и отображения счета в теннисном матче."""

REQUIRED_ADVANTAGE = 2


class TennisGame1:
    """Класс для отслеживания партии по теннису.

    Этот класс позволяет вести счет очков и
    выводить их.

    Attributes:
        player1_name (str): Имя первого игрока.
        player2_name (str): Имя второго игрока.
        p1points (int): Текущее количество очков у первого игрока (по умолчанию 0).
        p2points (int): Текущее количество очков у второго игрока (по умолчанию 0).
    """

    def __init__(self, player1_name: str, player2_name: str) -> None:
        """Инициализирует объект теннисный матч."""

        self.player1_name = player1_name
        self.player2_name = player2_name
        self.p1points = 0
        self.p2points = 0

    def add_point(self, player_name: str) -> None:
        """Начисляет очко игроку.

        Args:
            player_name (str): Имя игрока.
        """

        if player_name == self.player1_name:
            self.p1points += 1
        else:
            self.p2points += 1

    def display_tie_score(self) -> str:
        """Выводит счет при равенстве очков.

        Returns:
            str: Очки игроков.
        """

        result = ""
        if self.p1points == self.p2points:
            result = {
                0: "Love-All",
                1: "Fifteen-All",
                2: "Thirty-All",
            }.get(self.p1points, "Deuce")
        return result

    def dispaly_endgame_score(self) -> str:
        """Выводит счет при преимуществе или победе одного из игроков.

        Returns:
            str: Очки игроков.
        """

        result = ""
        minus_result = self.p1points - self.p2points
        match minus_result:
            case 1:
                result = "Advantage player1"
            case -1:
                result = "Advantage player2"
            case x if x >= REQUIRED_ADVANTAGE:
                result = "Win for player1"
            case _:
                result = "Win for player2"
        return result

    def display_score(self) -> str:
        """Выводит текущий счет.

        Returns:
            str: Очки игроков.
        """

        score_names = {
            0: "Love",
            1: "Fifteen",
            2: "Thirty",
            3: "Forty",
        }

        p1_score = score_names[self.p1points]
        p2_score = score_names[self.p2points]

        result = f"{p1_score} - {p2_score}"
        return result


if __name__ == "__main__":
    game = TennisGame1("Игорь", "Сергей")
    game.add_point("Сергей")
    points = game.display_score()
    print(f"Счет: {points}")
