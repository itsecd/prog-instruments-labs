#!/usr/bin/env python
"""
Python Music Player
By Benjamin Urquhart
VERSION: 2.4.3

This player is designed to play music on a Raspberry Pi,
but can be used on Windows and OSX. OSX support is limited.

Don't expect good documentation for a little while.
"""

__all__ = []
__version__ = '2.4.3'
__author__ = 'Benjamin Urquhart'

import datetime
import os
import sys
import string
import tarfile
import traceback
import urllib
import urllib2
from random import randint
from threading import Thread as Process
from time import sleep
from typing import List

try:
    import requests
except ImportError:
    pass

thread_use: bool = False
stop: bool = False
skip: bool = False
pause: bool = False
play: bool = False
debug: bool = False
option: str = "n"
select: int = 0
current_song: str = ""
amount: int = 0
played_songs: List[str] = []
playlist: List[str] = []
check: str = ""
width: int = 800
height: int = 600
console: bool = False
text: str = ''
song_num: int = 1
kill: bool = False

print("Starting Python Music Player " + __version__)

def mkdir(directory: str) -> None:
    """Create a directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def touch(path: str) -> None:
    """Create a file or update its timestamp."""
    with open(path, 'a'):
        os.utime(path, None)


def shutdown() -> None:
    """Shutdown the player gracefully."""
    try:
        bcast("\n")
        bcast("Stopping...")
        pygame.mixer.music.stop()
        log("Shutdown success")
        log_file.close()
        pygame.quit()
        quit()
    except:
        log("An error occurred")
        log_error()
        log_file.close()
        pygame.quit()
        quit()


def log(message: str) -> None:
    """Log a message to the log file."""
    try:
        if debug:
            print("[Debug]: " + message)
        log_file.write("[Logger]: ")
        log_file.write(message)
        log_file.write("\n")
    except:
        pass


def log_error() -> None:
    """Log the last error to the log file."""
    try:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        log('')
        log(''.join(line for line in lines))
        if debug:
            bcast("[Error]: " + ''.join(line for line in lines), True)
    except:
        pass


def bcast(message: str, err: bool = False) -> None:
    """Broadcast a message to the console or log."""
    try:
        if err:
            print(message)
        else:
            print("[Player]: " + message)
    except:
        pass


def updater() -> None:
    """Check for updates and install if available."""
    log('Update requested; attempting...')
    if update == 0:
        bcast('No update found.')
        log('No update found')
    else:
        bcast('Attempting to retrieve tarball...')
        try:
            log('Connecting to ' + url + '...')
            try:
                r = requests.get('http://' + url)
                status = r.status_code
            except:
                status = 200
                log_error()
            if status == 200:
                try:
                    filename = urllib.urlretrieve(
                        'http://' + url + '/python/downloads/player/'
                        'music-player-' + str(ver) + '.tar.gz',
                        'music-player-' + str(ver) + '.tar.gz')
                except:
                    log_error()
                    raise IOError
                bcast('Installing...')
                log('Download success')
                log('Will now attempt to install update')
                try:
                    mkdir('update')
                    os.rename("music-player-" + str(ver) + ".tar.gz",
                              'update/update.tar.gz')
                    os.chdir('update')
                    tar = tarfile.open("update.tar.gz")
                    tar.extractall()
                    tar.close()
                    os.remove('update.tar.gz')
                    os.chdir('..')
                    log('Success!')
                    bcast('Done!')
                    bcast("Move 'player.py' from the folder 'update' to: " +
                          os.path.dirname(os.getcwd()))
                except:
                    log_error()
                    bcast('Installation failed')
            else:
                bcast('Server is down')
                raise IOError
        except:
            log_error()
            bcast('Download failed')


def server() -> None:
    """Start a control server for the player."""
    try:
        import socket
        host = socket.gethostname()
        port = 9000
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((host, port))
        except socket.error as msg:
            print('Bind failed. Error Code : ' + str(msg[0]) +
                  ' Message ' + msg[1])
            log_error()
            try:
                s.close()
            except:
                log_error()
                pass
        s.listen(2)
        print('Started control server on ' + host + ':' + str(port))
    except:
        print("Couldn't create control server")
        log_error()


def news() -> None:
    """Fetch and display news updates."""
    log("Getting news")
    try:
        news_response = urllib2.urlopen("http://" + url + "/news.txt")
        news_content = news_response.read()
        if news_content == '':
            bcast("No News")
            log("No News")
        else:
            bcast(news_content)
            log(news_content)
    except:
        log_error()
        bcast("Couldn't get news updates", True)


def control() -> None:
    """Handle user input for controlling the player."""
    global thread_use, option
    thread_use = True
    option = ''
    option = raw_input('> ')
    try:
        option = option.replace("\n", '')
        option = option.lower()
        if option in ('quit', 'stop'):
            print("Use Control-C to quit")
        elif option == 'skip':
            pygame.mixer.music.stop()
        elif option == 'update':
            updater()
        elif option == 'pause':
            pygame.mixer.music.pause()
            bcast('Paused')
        elif option == 'play':
            pygame.mixer.music.play()
        elif option == '':
            option = ''
        elif option == 'debug':
            debug = not debug
            print("Debug mode " + ("enabled" if debug else "disabled"))
        elif option == "news":
            news()
        else:
            bcast("Invalid command: " + option)
    except:
        log_error()
    sleep(0.1)
    thread_use = False


def control2() -> None:
    """Handle keyboard events for controlling the player."""
    try:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    print("Debug")
                    debug = not debug
                if event.key in (pygame.K_SPACE, pygame.K_F11):
                    bcast("Pause")
                    if pause:
                        pygame.mixer.music.play()
                        pause = False
                    else:
                        pygame.mixer.music.pause()
                        pause = True
                if event.key == pygame.K_u:
                    bcast("Update")
                    updater()
                if event.key == pygame.K_F12:
                    bcast("Skip")
                    pygame.mixer.music.stop()
                if event.key in (pygame.K_F10, pygame.K_q):
                    bcast("Quit")
                    shutdown()
    except:
        log_error()
    sleep(0.2)

mkdir('logs')
current_time = datetime.datetime.now()
try:
    log_file = open("./logs/" + str(current_time), "w+")
except:
    log_error()
    bcast("Failed to create log")


def display(text: str, background: 'pygame.Surface', 
            screen: 'pygame.Surface') -> None:
    """Display text on the screen."""
    font = pygame.font.Font("freesansbold", 36)
    out = font.render(text, 1, (10, 10, 10))
    text_pos = out.get_rect()
    text_pos.centerx = background.get_rect().centerx
    background.blit(out, text_pos)
    screen.blit(background, (0, 0))
    pygame.display.flip()

# server()

try:
    import pygame
    from pygame.locals import *
except ImportError:
    log_error()
    try:
        print("Downloading assets")
        log('Pygame missing; getting installer')
        os_version = sys.platform
        if os_version == 'win32':
            urllib.urlretrieve(
                'https://pygame.org/ftp/pygame-1.9.1.win32-py2.7.msi',
                'pygame-1.9.1.msi')
        elif os_version == 'darwin':
            urllib.urlretrieve(
                'https://pygame.org/ftp/pygame-1.9.1release-python.org-'
                '32bit-py2.7-macosx10.3.dmg', 'pygame-mac.dmg')
            log('Success!')
        elif os_version in ('linux2', 'cygwin'):
            print('You are using linux or cygwin')
            print("Use the command 'sudo pip install pygame' to download\n"
                  "the necessary modules")
            log(os_version + ' detected; pip installer recommended')
        else:
            print('Unrecognized os: ' + os_version)
        try:
            urllib.urlretrieve('http://' + url + '/pygame.tar.gz',
                               'pygame.tar.gz')
            tar = tarfile.open("pygame.tar.gz")
            tar.extractall()
            tar.close()
            os.remove('pygame.tar.gz')
        except:
            log_error()
            print("Failed to get assets")
            exit()
        print('Please run the installer that has been dropped into the ' +
              os.path.dirname(os.getcwd()) + ' folder')
    except:
        print('Failed to get assets')
        print("Please install the 'pygame' module manually at pygame.org")
        log_error()
        shutdown()
    exit()

try:
    pygame.init()
    pygame.mixer.init()
    log('Pygame initialized')
except:
    bcast("Couldn't run pygame.init()", True)
    log("pygame.init() failed")
    log_error()

try:
    if len(sys.argv) > 1:
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg in ("--console", "-c"):
                console = True
            elif arg in ("--verbose", "-v"):
                debug = True
            elif arg in ("-f", "--file"):
                pygame.init()
                try:
                    pygame.mixer.music.load(sys.argv[i + 1])
                    print("Now Playing: " + sys.argv[i + 1])
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        continue
                    kill = True
                except:
                    log_error()
                    print("There was an error playing the file")
                    kill = True
            elif arg in ("-h", "--help"):
                print('Plays music in the "Music" folder within the current '
                      'directory\n')
                print("Usage: " + sys.argv[0] + " [-hvc] [-f <filepath>]")
                print("Options: ")
                print("\t -h, --help\t Displays this help text")
                print("\t -v, --verbose\t Displays extra information")
                print("\t -c, --console\t Disables Pygame screen (text-only mode)")
                print("\t -f, --file\t Plays the file at the filepath specified")
                print("\nExamples: \n\t " + sys.argv[0] +
                      " -v -c -f /sample/file/path/foo.bar")
                print("\t " + sys.argv[0] + " -f foo.bar")
                kill = True
            i += 1
except:
    pass
if kill:
    exit()

url: str = "benjaminurquhart.me"
update: int = 0
try:
    log('Checking for updates...')
    log('Getting info from ' + url)
    ver = urllib2.urlopen('http://' + url + '/version.txt')
    rev = urllib2.urlopen('http://' + url + '/rev.txt')
    ver = ver.read()
    rev = rev.read()
    if float(ver) > float(__version__):
        log('Update found!')
        bcast("Python Music Player " + ver + " is available")
        bcast("Type update at the prompt to download")
        update = 1
    elif float(ver) < float(__version__):
        log('Indev version in use')
        bcast('Indev version in use')
    elif int(rev) > int(revision) and float(ver) == float(__version__):
        log('New revision found!')
        bcast('Revision ' + str(rev) + ' is available')
        bcast('Type update at the prompt to download')
        update = 1
    elif float(ver) == float(__version__):
        log('No update found')
        bcast('No update found')
except:
    bcast('Failed to check for updates', True)
    log_error()
    log('Update check failed')

mkdir('Music')
log("Player starting...")
news()

try:
    if not console:
        screen = pygame.display.set_mode((1000, 200))
        pygame.display.set_caption("Music Player")
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((250, 250, 250))
except:
    bcast("Display error, console mode active", True)
    log("Display error")
    log_error()
    console = True
log("Player started")
sound_data = os.listdir('./Music')
try:
    for i in sound_data:
        playlist.append(i)
    i = 0
    amount = len(playlist)
    log(str(amount) + " Songs found")
    if amount == 0:
        bcast('No music found!')
        shutdown()
    bcast("Number of songs: " + str(amount))
    while i != amount:
        select = randint(0, amount - 1)
        if option.lower() == "y":
            for i in song:
                current_song = i
            option = "n"
        else:
            current_song = playlist[select]
        if current_song not in played_songs:
            bcast("Now Playing: " + current_song + " (" + str(song_num) +
                  " out of " + str(amount) + ")")
            log("Song " + str(song_num) + " out of " + str(amount))
            try:
                log("Loading '" + current_song + "'")
                pygame.mixer.music.load("./Music/" + current_song)
                log('Now Playing: ' + current_song)
            except:
                bcast("Couldn't play " + current_song)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if not console:
                    screen.blit(background, (0, 0))
                    control2()
                else:
                    t = Process(None, control())
                    t.daemon = False
                    t.start()
            if current_song not in played_songs:
                played_songs.append(current_song)
                i += 1
            sleep(0.2)
            song_num += 1
    bcast("All songs have been played!")
    log('All songs have been played')
    shutdown()
except:
    log_error()
    shutdown()
