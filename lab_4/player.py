#!/usr/bin/python

# ---------------- READ ME ---------------------------------------------
# This Script is Created Only For Practise And Educational Purpose Only
# This is an Example Of Tkinter Canvas Graphics
# This Script Is Created For http://bitforestinfo.blogspot.in
# This Script is Written By
#
#
##################################################
######## Please Don't Remove Author Name #########
############### Thanks ###########################
##################################################
#
#
__author__='''

######################################################
                By S.S.B Group                          
######################################################

    Suraj Singh
    Admin
    S.S.B Group
    surajsinghbisht054@gmail.com
    http://bitforestinfo.blogspot.in/

    Note: We Feel Proud To Be Indian
######################################################
'''
# Here Importing Modules
import logging
import pyglet     # import pyglet
import datetime
import threading

import pyglet.media as media

from Configuration_base import *

# ============================================
# Usages:
#       player=__media__player(path, song_time, song_duration, volume)
# Here:
#   path=String Variable For Song Path
#   song_time=String Variable For Song Playing Time
#   song_duration= String Variable For Time duration
#   volume = IntVar For Volume Updates
#
# For Other Functions:
#   player.YourFunction
# ============================================

logging.basicConfig(
    level=logging.DEBUG,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler("lab_4/mediaplayer.log", mode='a', encoding='utf-8')  
    ]
)


class mediaplayer:

    def __init__(self, path, song_time, song_duration, volume):
        self.path=path                      # Song Playing Song
        self.volume=volume                  # Song Volume Update
        self.songtime=song_time             # Song Time Variable
        self.songduration=song_duration     # Song Duration
        self.player=media.Player()          # pyglet Media Player
        self.player.volume=1.5              # 
        self.time_thread()                  # Time Updating Thread

        self.path.trace('w', self.play_song)
        self.volume.trace('w', self.volume_)
        logging.info("mediaplayer initialized with default volume: %.1f", self.player.volume)
        
    def jump(self, time):
        try:
            self.player.seek(time)
            logging.info("Jumped to time: %.2f seconds", time) 
        except Exception as e:
            print ('[+] Jump is Not Possible')
            return
           
    def now(self):
        storeobj=self.player.time
        return storeobj
    
    def now_(self):
        time=int(self.now())
        k=datetime.timedelta(seconds=time)
        k=k.__str__()
        return k
   
    def pause(self):
        self.player.pause()    
        logging.info("Playback paused.") 

    def play(self):
        self.player.play()
        logging.info("Playback started.")
    
    def stop(self):
        self.reset_player()
        logging.info("Playback stopped.")
    
    def volume_(self, *args, **kwargs):
        try:
            volume=self.volume.get()
            self.player.volume=volume
            logging.info("Volume set to: %.2f", volume)
        except Exception as e:
            pass
        return
    
    def time_thread(self):
        threading.Thread(target=self.update_time_).start()
    
    def update_time_(self):
        while True:
            now=self.now_()
            try:
                self.songtime.set(now)
            
            except Exception as e:
                print (e)
    
    def duration(self):
        try:
            storeobj=self.player.source.duration
            return storeobj
        except Exception:
            return '0'
        
    def duration_(self):
        time=self.duration()+10.0
        k=datetime.timedelta(seconds=time)
        k=k.__str__()
        return k
    
    def reset_player(self):
        self.player.pause()
        self.player.delete()
        logging.info("Player reset.")
            
    def play_song(self, *args, **kwargs):
        if self.path.get():
            try:
                self.reset_player()
                logging.info("Attempting to play song from path: %s", self.path.get())
                try:
                    src=media.load(self.path.get())
                    self.player.queue(src)
                    self.play()
                    
                    self.songduration.set(self.duration_())   # Updating duration Time
                    logging.info("Song started playing. Duration: %s", self.songduration.get())
                except Exception as e:
                    print ("[+] Something wrong when playing song", e)
                    return 
            except Exception as e:
                print (' [+] Please Check Your File Path', self.path.get())
                print (' [+] Error: Problem On Playing \n ', e)
                return 
        else:
            print (' [+] Please Check Your File Path', self.path.get())
        return

    def fast_forward(self):
        time = self.player.time + jump_distance
        try:
            if self.duration() > time:
                self.player.seek(time)
                logging.info("Fast-forwarded to %.2f seconds", time)
            else:
                self.player.seek(self.duration())
                logging.info("Fast-forwarded to the end of the song.")
        except AttributeError:
            pass

    def rewind(self):
        time = self.player.time - jump_distance
        try:
            self.player.seek(time)
        except Exception as e:
            self.player.seek(0)
            
 