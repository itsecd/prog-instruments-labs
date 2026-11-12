from GameOfLife_utils import get_history, make_movie, plotcells, read_rle

""" Initial board """

## Random board 60x60 with 20% alive
# board = np.random.choice(2,(60,60),p=[0.8,0.2]).astype(bool)

################################


def do_it(pattern, shape_y, pos, steps, trim=False, rev_h=False, rev_v=False, tp=False):
    """Load a pattern from an RLE file, run evolution and make a movie
    Options :
    * pos = where to position the pattern on the board
    * rev_h, rev_v and tp to reverse horizontal, vertical or transpose
      the dimensions
    * trim : trim edges for nicer plot
    """
    # 16/9 ratio
    shape = (int(1.78 * shape_y), shape_y)

    # Read RLE file in the rle folder
    board = read_rle(
        "rle/" + pattern + ".rle", shape, pos, rev_h=rev_h, rev_v=rev_v, tp=tp
    )
    history = get_history(board, steps)

    plotcells(history[0, :, :], "output/" + pattern + "_init.png")
    plotcells(history[-1, :, :], "output/" + pattern + "_end.png")
    make_movie(history, "output/" + pattern + ".mp4", trim=trim)


################################

# pattern = "example1"
# shape_y = 10
# pos = (6,4)
# steps = 15
# do_it(pattern,shape_y,pos,steps)

# pattern = "blinker"
# shape_y = 9
# pos = (6,4)
# steps = 30
# do_it(pattern,shape_y,pos,steps)

# pattern = "four"
# shape_y = 9
# pos = (6,4)
# steps = 10
# do_it(pattern,shape_y,pos,steps)

# pattern = "five"
# shape_y = 18
# pos = (13,8)
# steps = 20
# do_it(pattern,shape_y,pos,steps)

# pattern = "example2"
# shape_y = 10
# pos = (7,4)
# steps = 15
# do_it(pattern,shape_y,pos,steps)

# pattern = "stills"
# shape_y = 24
# pos = (8,2)
# steps = 4
# do_it(pattern,shape_y,pos,steps)

# pattern = "oscillos"
# shape_y = 16
# pos = (4,2)
# steps = 30
# do_it(pattern,shape_y,pos,steps)

# pattern = "beehiveplus"
# shape_y = 20
# pos = (15,7)
# steps = 30
# do_it(pattern,shape_y,pos,steps)

# pattern = "stairs6"
# shape_y = 28
# pos = (22,13)
# steps = 100
# do_it(pattern,shape_y,pos,steps)

pattern = "Rpento"
shape_y = 100
pos = (70, 55)
steps = 1500
do_it(pattern, shape_y, pos, steps, trim=True)

# pattern = "cthulhu"
# shape_y = 17
# pos = (9,2)
# steps = 10
# do_it(pattern,shape_y,pos,steps,rev_v=True)

# pattern = "exploder"
# shape_y = 20
# pos = (15,7)
# steps = 100
# do_it(pattern,shape_y,pos,steps)

# pattern = "ten"
# shape_y = 15
# pos = (8,7)
# steps = 100
# do_it(pattern,shape_y,pos,steps)

# pattern = "104P177"
# shape_y = 70
# pos = (40,12)
# steps = 1000
# do_it(pattern,shape_y,pos,steps)

# pattern = "spaceships"
# shape_y = 40
# pos = (60,9)
# steps = 130
# do_it(pattern,shape_y,pos,steps,rev_h=True)

# pattern = "canadagoose"
# shape_y = 50
# pos = (50,35)
# steps = 150
# do_it(pattern,shape_y,pos,steps,rev_v=True,rev_h=True)

# pattern = "60P5H2V0"
# shape_y = 50
# pos = (3,16)
# steps = 200
# do_it(pattern,shape_y,pos,steps,rev_v=True,rev_h=True, tp=True)

# pattern = "puffer1"
# shape_y = 150
# pos = (5,65)
# steps = 600
# do_it(pattern,shape_y,pos,steps,rev_v=True,tp=True)

# pattern = "backrake3"
# shape_y = 200
# pos = (5,90)
# steps = 500
# do_it(pattern,shape_y,pos,steps,rev_v=True,tp=True, trim=True)

# pattern = "linepuffer"
# shape_y = 300
# pos = (50,65)
# steps = 800
# do_it(pattern,shape_y,pos,steps,rev_v=True,tp=True)

# pattern = "gosperglidergun"
# shape_y = 60
# pos = (5,5)
# steps = 200
# do_it(pattern,shape_y,pos,steps)

# pattern = "10cellinfinitegrowth"
# shape_y = 360
# pos = (400,300)
# steps = 4000
# do_it(pattern,shape_y,pos,steps,trim=True)

# pattern = "breeder1"
# shape_y = 1080
# pos = (10,400)
# steps = 2500
# do_it(pattern,shape_y,pos,steps,rev_v=True, trim=True )

# pattern = "max"
# shape_y = 720
# pos = (640,360)
# steps = 1000 # 1000
# do_it(pattern,shape_y,pos,steps, trim=True)

# pattern = "p41660p5h2v0gun"
# shape_y = 1200
# pos = (500,100)
# steps = 5000 # 1000
# do_it(pattern,shape_y,pos,steps, trim=True)

# pattern = "otcametapixeloff"
# shape_y = 2200
# pos = (800,50)
# steps = 1000 # 1000
# do_it(pattern,shape_y,pos,steps)


## Random 20% alive
# for s in range(100):
#    np.random.seed(s)
#    board = np.random.choice(2,(3*16,3*9),p=[0.85,0.15]).astype(bool)
#    steps = 300
#    history = get_history(board,steps)
#    #plotcells(history[0,:,:],"output/random"+str(s)+"_init.png")
#    plotcells(history[-1,:,:],"output/random"+str(s)+"_end.png")
#    #make_movie(history,"output/random"+str(s)+".mp4")

# s = 1
# np.random.seed(s)
# board = np.random.choice(2,(3*16,3*9),p=[0.85,0.15]).astype(bool)
# steps = 100
# history = get_history(board,steps)
# plotcells(history[0,:,:],"output/random"+str(s)+"_init.png")
# plotcells(history[-1,:,:],"output/random"+str(s)+"_end.png")
# make_movie(history,"output/random"+str(s)+".mp4")
