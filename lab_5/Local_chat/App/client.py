import socket
import threading
import time

import multiprocessing 
import tkinter as tk

from App.logger import ChatLogger


class Client():
    """
    This class represents the chat client. 
    Handles connection to the server, sending
    and receiving messages, and GUI updates.
    """

    def __init__(self, ip_adr, port, key, nickname, queue, queue_send):
        """
            Args:
                ip_adr (str): The IP address of the server to connect to.
                port (int): The port number of the server.
                key (str): The private key used for authentication.
                nickname (str): The nickname of the client (user).
                queue (multiprocessing.Queue): Queue for incoming
                messages from the server.
                queue_send (multiprocessing.Queue): Queue for outgoing messages
                to be sent to the server.
        """
        self.ip_adr = ip_adr
        self.port = port 
        self.key = key
        self.nickname = nickname
        self.socket = None
        self.root = None
        self.label1 = None
        self.button1 = None
        self.scrollbar = None
        self.text_output = None
        self.full_recieved_msg = ''
        self.queue = queue
        self.queue_send = queue_send
        self.logger = ChatLogger().getLogger(self.__class__.__name__)

    def connect_to_server(self):
        """
        Attempts to connect to the specified server.
        This method creates a socket connection to the
        server using the provided IP address and port.
        It retries the connection up to five times in case of failure.
        Returns:
            bool: Returns True if the connection was successful,
            False if the connection failed after five attempts.
        """
        self.socket = socket.socket()
        count_of_connection = 0
        while True:
            try:
                self.socket.connect((self.ip_adr, self.port))
                break
            except socket.error as error:
                self.logger.warning("Error while connecting to server %s" %error)
                count_of_connection += 1
                if count_of_connection > 4:
                    self.logger.warning("You try it for 5+ times"\
                                        "we gonna close your connection")
                    self.socket.close()
                    return False
                time.sleep(1)
        list_for_join = []
        nickname_enc = self.nickname.encode('utf-8')
        need_bytes_of_zero = 16 - len(nickname_enc)

        list_for_join.append(b'\x00'*need_bytes_of_zero)
        list_for_join.append(nickname_enc)

        msg_to_send = b''.join(list_for_join)

        try:
            self.socket.send(msg_to_send)
        except socket.error as error:
            self.logger.error("Sorry, we can't send your message")
        return True
    
    def send_msg(self):
        """Sends messages from the queue to the server."""
        while True:
            if not self.queue_send.empty():
                kboard_input = self.queue_send.get()
                list_for_join = []

                message_enc = kboard_input.encode("utf-8")

                nickname_enc = self.nickname.encode('utf-8')
                need_bytes_of_zero = 16 - len(nickname_enc)

                list_for_join.append(message_enc)
                list_for_join.append(b'\x00' * need_bytes_of_zero)
                list_for_join.append(nickname_enc)

                msg_to_send = b''.join(list_for_join)
                self.logger.info("ALL_SEND: %s: %s" %(self.nickname, kboard_input))
                try:
                    self.socket.send(msg_to_send)
                except socket.error as error:
                    self.logger.warning("Sorry, we can't send your message \t %s" %error)

    def receive_msg(self):
        """
        Receives messages from the server
        and puts them into the queue.
        """
        while True:
            recieved_msg = self.socket.recv(128)
            
            if recieved_msg[0] == 194:
                recieved_line = recieved_msg.decode("utf-8")
            else:
                recieved_line = recieved_msg.decode("utf-8")
                nickname = recieved_line[-16::].replace("\x00", "")                

                if nickname != self.nickname.replace("\x00", ""):
                    message = recieved_line[0:-16]
                    self.full_recieved_msg = f"{nickname}: {message}"
                    self.queue.put(self.full_recieved_msg)    

    def add_lines(self):
        """
        Adds received messages
        from the queue to the text output.
        """
        try:         
            if not self.queue.empty():
                recieved_msg_from_queue = self.queue.get()
                kastil = "".join(map(
                    str, list(recieved_msg_from_queue))).replace('\x00', '')

                self.text_output.insert("end", kastil + "\n") 
                self.text_output.see("end")  
            self.root.after(100, self.add_lines)  
        except Exception as ex:
            self.logger.error("An unexpected error occurred: %s" %ex)
    
    def send_msg_button(self):
        """
        Handles sending a message when
        the send button is pressed.
        """
        msg_for_send = self.entry1.get()
        self.queue.put(f'{self.nickname} : {msg_for_send}')
        self.queue_send.put(msg_for_send)
        self.entry1.delete(0, 'end')

    def run_gui(self):
        """Initialize and runs the GUI in mainloop."""
        self.root= tk.Tk()

        self.label1 = tk.Label(self.root, text='Anon chat')
        self.label1.config(font=('helvetica', 14))
        self.label1.place(x=220, y=15)

        self.entry1 = tk.Entry(self.root) 
        self.entry1.place(x=15, y=400, width=450, height=50)

        self.button1 = tk.Button(self.root, text='send',
                                 command=self.send_msg_button)
        self.button1.place(x=400, y=450)

        self.scrollbar = tk.Scrollbar(self.root)
        self.scrollbar.pack(side="right", fill="none", expand=True)
        self.text_output = tk.Text(self.root, 
                                   yscrollcommand=self.scrollbar.set)
        self.text_output.place(x=15, y=50, width=450, height=300)
        self.scrollbar.config(command=self.text_output.yview)

        self.root.minsize(500, 500)
        self.root.maxsize(500, 500)

        self.root.after(0, self.add_lines)
        self.root.mainloop()

    def run(self):
        """
        Runs the client, starting threads for GUI,
        sending, and receiving messages.
        """                              
        GUITread = multiprocessing.Process(target=self.run_gui)
        send_thread = threading.Thread(target=self.send_msg)
        receive_thread = threading.Thread(target=self.receive_msg)

        GUITread.start()
        send_thread.start()
        receive_thread.start()       

        send_thread.join()
        receive_thread.join()                        

    def close_conn(self):
        """Closes the client's socket connection."""
        self.socket.close()