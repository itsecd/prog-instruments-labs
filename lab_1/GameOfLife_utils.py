#!/usr/bin/env python3

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Path of ffmpeg executable for animation
plt.rcParams["animation.ffmpeg_path"] = r"/Volumes/Data/Youtube/[ffmpeg]/ffmpeg"

# Plot settings
LINE_WIDTH = 0.5
MOVIE_FIGSIZE = (16, 9)
MOVIE_DPI = 240

# Boards bigger than this are drawn with imshow instead of pcolor
IMSHOW_THRESHOLD = 200

# Maximal size of a pattern read from an RLE file
SHAPE_MAX = (2500, 2500)

###############################################################################


def evolve(board):
    """Evolves a board of Game of Life for one turn"""
    # Dead cells as a boundary condition
    # Count neighbours
    # Alive if 3 neighbours or 2 neighbours and already alive

    xi = board.astype(int)
    neigh = np.zeros(xi.shape)
    neigh[1:-1, 1:-1] = (
        xi[:-2, :-2]
        + xi[:-2, 1:-1]
        + xi[:-2, 2:]
        + xi[1:-1, :-2]
        + xi[1:-1, 2:]
        + xi[2:, :-2]
        + xi[2:, 1:-1]
        + xi[2:, 2:]
    )

    return np.logical_or(neigh == 3, np.logical_and(xi == 1, neigh == 2))


###############################################################################


def get_history(board, steps):
    """Returns the evolution of a board after the given number of generations"""
    history = np.zeros((steps, board.shape[0], board.shape[1]), dtype=bool)
    for t in range(steps):
        history[t, :, :] = board
        board = evolve(board)
        print(t)
    return history


###############################################################################


def plotcells(board, filename=None):
    """Plots a board of Game of Life + optionally saving the figure"""
    use_imshow = board.shape[0] > IMSHOW_THRESHOLD

    fig = plt.figure(figsize=(16, 9), dpi=120)
    if not use_imshow:
        # Light blue lines as cells boundaries
        plt.pcolor(
            board.T, cmap="gray_r", edgecolors="cadetblue", linewidths=LINE_WIDTH
        )
    else:
        plt.imshow(board[:, ::-1].T, cmap="gray_r")
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    fig.tight_layout()

    if filename:
        plt.savefig(filename, dpi=90)
    else:
        plt.show()


###############################################################################


def make_movie(history, filename, trim=False):
    """Create the movie from a history of a game of life"""
    # History is the boolean history (non inverted i.e. True = alive)
    # Inversion is done in the colormap
    # Filename should be *.mp4

    use_imshow = history.shape[1] > IMSHOW_THRESHOLD

    # Trim boundaries
    if trim:
        history = history[:, 3:-3, 3:-3]

    # Create the plot and its starting point
    print("Create initial plot")
    my_cmap = plt.get_cmap("gray_r")
    fig = plt.figure(figsize=MOVIE_FIGSIZE, dpi=MOVIE_DPI)
    ax = fig.add_subplot(111)

    if not use_imshow:
        # First option : use pcolor
        pc = ax.pcolor(
            history[0, :, :].T,
            cmap=my_cmap,
            edgecolors="cadetblue",
            linewidths=LINE_WIDTH,
        )
    else:
        # Second option : use imshow
        im = ax.imshow(history[0, :, ::-1].T, cmap=my_cmap)

    cnt = ax.text(
        0.01,
        0.99,
        str(0),
        color="red",
        fontsize=30,
        verticalalignment="top",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()

    # The function as it is called at the n-th iteration
    # It directly modifies the data within the image
    def update_img(n):
        # Revert and scale from 0-1 to 0-255
        print("Frame " + str(n))
        if not use_imshow:
            new_color = my_cmap(255 * history[n, :, :].T.ravel())
            pc.update({"facecolors": new_color})
        else:
            im.set_data(history[n, :, ::-1].T)
        #
        cnt.set_text(str(n))
        #    # if needed, can modify the field of view
        #    fov =
        #    ax.set_xlim()
        #    ax.set_ylim()

        return True

    # Create the animation and save it
    print("Make animation")
    ani = animation.FuncAnimation(
        fig, update_img, history.shape[0], interval=30
    )  # 30ms per frame
    writer = animation.FFMpegWriter(fps=30, bitrate=5000)
    print("Save movie")
    ani.save(filename, writer=writer, dpi=MOVIE_DPI)
    print("Saved")


###############################################################################


def read_rle(
    filename, cshape=(50, 50), position=(10, 10), rev_h=False, rev_v=False, tp=False
):
    """Read the RLE file and returns a binary matrix"""
    # see http://www.conwaylife.com/wiki/RLE

    # Open file and cast it into a unique string
    s = ""
    with open(filename) as f:
        for line in f:
            if line[0] in "#x":
                continue
            s += line[:-1]  # To remove EOL

    # Create matrix
    board = np.zeros(SHAPE_MAX, dtype=bool)

    # We parse each character and decide accordingly what to do
    # If the character is a digit, we keep going until we reach 'b' or 'o'
    cur_x, cur_y = 0, 0
    qs = ""
    for c in s:
        # Next Line
        if c == "$":
            q = 1 if qs == "" else int(qs)
            cur_y += q
            cur_x = 0
            qs = ""

        # Digit
        if "0" <= c <= "9":
            qs += c

        # Alive (o) or Dead (b) cell
        if c in "bo":
            q = 1 if qs == "" else int(qs)
            for _ in range(q):
                board[cur_x, cur_y] = c == "o"
                cur_x += 1
            qs = ""

    pos_x, pos_y = position
    size_y = np.where(board.sum(axis=0) > 0)[0].max() + 1
    size_x = np.where(board.sum(axis=1) > 0)[0].max() + 1

    board = board[0:size_x, 0:size_y]

    if rev_v:
        board = board[:, ::-1]
    if rev_h:
        board = board[::-1, :]
    if tp:
        board = board.T

    result = np.zeros(cshape)
    result[pos_x : (pos_x + board.shape[0]), pos_y : (pos_y + board.shape[1])] = (
        np.copy(board)
    )

    return result.astype(bool)
