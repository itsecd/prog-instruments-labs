def initialize(window):
    global board
    if len(sys.argv) > 2:
        if int(sys.argv[1]) < 20 or int(sys.argv[2]) < 50:
            curses.endwin()
            sys.stderr.write("\nError[1] : minimum val for row,column is 20,50 respectively\n")
            return
        else:
            board = GameBoard(int(sys.argv[1]), int(sys.argv[2]))
    else:
        board = GameBoard(35, 70)
    game = True
    while game:
        window.clear()
        window.insstr(0, 0, str(board))
        window.refresh()
        status = True
        control = Thread(target=controller, args=(board, window))
        control.start()
        while status:
            window.clear()
            window.insstr(0, 0, str(board))
            window.refresh()
            speed = (10 / (len(board.snake) + 8))
            time.sleep(speed)
            status = board.refreshBoard()
        (game, board) = gameOver(window, board)
    curses.endwin()
    print("""
		Thank You for playing
		Credits:
			Name  : Aditya Jain
			Email : adityajn105@gmail.com
			Github: https://github.com/Adityajn
		""")