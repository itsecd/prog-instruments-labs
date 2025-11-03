class GameBoard(object):
    def __init__(self, row, col, size=2):
        self.size = size
        self.col = col
        self.row = row

        self.finished = False

        # inititalize empty board
        self.board = [[" " for _ in range(col)] for _ in range(row)]

        # initialize snake positions
        head = (int(row / 2), int(col / 2))
        self.snake = [((head[0] + i), head[1]) for i in range(size)]

        # put food
        self.food = []
        food = Thread(target=self.putFood)
        food.start()

        # 1- up, 2-right, 3-down, 4- left
        self.move = 4