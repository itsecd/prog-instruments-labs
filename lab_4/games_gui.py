import tkinter as tk
from tkinter import messagebox, simpledialog
import random
# NOTE: We'll put your existing game logic into this file later, but we need to
# refactor them to use GUI elements instead of 'print' and 'input'.
import json
import os

def load_hangman_words(filename="hangman_words.json"):
    """Loads a list of words from a JSON file using the script's absolute path."""
    
    # Use the same logic to find the absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filename)
    
    if not os.path.exists(full_path):
        print(f"Error: Hangman data file not found at {full_path}")
        return ["error"] # Return a default list so the game doesn't crash
    
    try:
        with open(full_path, 'r') as f:
            data = json.load(f)
            # Ensure the loaded data is a list of strings
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                return [word.lower() for word in data]
            else:
                print(f"Error: Hangman data file is not a list of strings: {full_path}")
                return ["error"]
    except Exception as e:
        print(f"An error occurred while reading the Hangman file: {e}")
        return ["error"]

def load_quiz_questions(filename="quiz_data.json"):
    """Loads quiz questions from a JSON file using the script's absolute path."""

    # 1. Get the directory of the currently executing script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Combine the script directory and the filename to create the full path
    full_path = os.path.join(script_dir, filename)

    if not os.path.exists(full_path):
        print(f"Error: Quiz data file not found at {full_path}")
        return []
    
    try:
        with open(full_path, 'r') as f:
            data = json.load(f)
            return data
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {filename}. Check the file format.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return []

def normalize_answer(answer):
    """Normalizes answer by removing spaces and converting written numbers to digits."""
    # Dictionary for written numbers to digits
    word_to_num = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    
    # Convert to lowercase and remove spaces
    normalized = answer.strip().lower().replace(" ", "")
    
    # Replace written numbers with digits
    for word, digit in word_to_num.items():
        normalized = normalized.replace(word, digit)
    
    return normalized

# --- Global Class to Manage the Application ---
class GameApp(tk.Tk):
    """The main application window."""
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Python Text-based Games Collection")
        self.geometry("600x500")

        # Container Frame: All other frames (pages) will be stacked on top of this.
        container = tk.Frame(self, bg="#90EE90") 
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        
        # Add all pages/frames to the dictionary
        for F in (MainMenu, MoreGamesMenu, MastermindGUI, WordScrambleGUI, BattleshipGUI, SlotMachineGUI, TextAdventureGUI, NumberGuessingGUI, WordGuessingGUI, RockPaperScissorsGUI, HigherOrLowerGUI, DiceRollingGUI, QuizSelectionGUI, QuizGameGUI, TicTacToeSelectionGUI, TicTacToeGUI, GameWIP):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # Put all frames in the same location (stacking them)
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainMenu")

    def show_frame(self, page_name):
        """Raises the desired frame to the front."""
        frame = self.frames[page_name]
        frame.tkraise()

# --- Main Menu Screen ---
class MainMenu(tk.Frame):
    """The starting screen with buttons for all games."""
    def __init__(self, parent, controller):
        bg="#90EE90"
        tk.Frame.__init__(self, parent, bg=bg) 
        self.controller = controller

        tk.Label(self, text="=== Python Text-based Games Collection ===", font=('Arial', 16, 'bold'), bg=bg).pack(pady=20)
        
        # List of all games to create buttons dynamically
        games = [
            ("1. Number Guessing Game", "NumberGuessingGUI"),
            ("2. Word Guessing (Hangman)", "WordGuessingGUI"),
            ("3. Rock-Paper-Scissors", "RockPaperScissorsGUI"),
            ("4. Higher or Lower (1-100)", "HigherOrLowerGUI"),
            ("5. Dice Rolling (Betting)", "DiceRollingGUI"),
            ("6. Quiz Game", "QuizSelectionGUI"),
            ("7. Tic-Tac-Toe", "TicTacToeSelectionGUI"),
        ]

        for text, frame_name in games:
            if frame_name:
                tk.Button(self, text=text, width=50, command=lambda name=frame_name: controller.show_frame(name)).pack(pady=5)
            else:
                tk.Button(self, text=text, width=50, state=tk.DISABLED, command=lambda: messagebox.showinfo("WIP", "Game not yet implemented!")).pack(pady=5)

        tk.Button(self, text="More Games", width=30, command=lambda: controller.show_frame("MoreGamesMenu")).pack(pady=10)
        tk.Button(self, text="Exit", width=30, command=controller.quit).pack(pady=20)

# --- More Games Menu Screen ---
class MoreGamesMenu(tk.Frame):
    """The screen with additional games."""
    def __init__(self, parent, controller):
        bg="#90EE90"
        tk.Frame.__init__(self, parent, bg=bg) 
        self.controller = controller

        tk.Label(self, text="=== More Games ===", font=('Arial', 16, 'bold'), bg=bg).pack(pady=20)
        
        # List of additional games
        more_games = [
            ("8. Mastermind (Bulls and Cows)", "MastermindGUI"),
            ("9. Word Scramble (Anagrams)", "WordScrambleGUI"),
            ("10. Battleship", "BattleshipGUI"),
            ("11. Slot Machine", "SlotMachineGUI"),
            ("12. Text Adventure (Escape Room)", "TextAdventureGUI"),
        ]

        for text, frame_name in more_games:
            if frame_name == "BattleshipGUI":
                tk.Button(self, text=text, width=50, command=lambda name=frame_name: [controller.winfo_toplevel().geometry("1000x900"), controller.show_frame(name)]).pack(pady=5)
            elif frame_name == "TextAdventureGUI":
                tk.Button(self, text=text, width=50, command=lambda name=frame_name: [controller.winfo_toplevel().geometry("900x750"), controller.show_frame(name)]).pack(pady=5)
            else:
                tk.Button(self, text=text, width=50, command=lambda name=frame_name: controller.show_frame(name)).pack(pady=5)

        tk.Button(self, text="Back to Menu", width=30, command=lambda: controller.show_frame("MainMenu")).pack(pady=20)

# --- Mastermind (Bulls and Cows) GUI Frame ---
class MastermindGUI(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#90EE90")
        self.controller = controller
        
        # Game State Variables
        self.secret_number = ""
        self.attempts = 0
        self.max_attempts = 10
        
        # --- Widgets Setup ---
        tk.Label(self, text="=== Mastermind (Bulls and Cows) ===", font=('Arial', 16, 'bold'), bg="#90EE90").pack(pady=10)
        
        tk.Label(self, text="Guess the 4-digit number with unique digits!", font=('Arial', 12), bg="#90EE90").pack(pady=5)
        
        # A Text widget to act as the Game Log
        self.log_text = tk.Text(self, height=8, width=60, state=tk.DISABLED)
        self.log_text.pack(pady=10)
        
        # Label to show status (Attempts left)
        self.status_label = tk.Label(self, text="", bg="#90EE90")
        self.status_label.pack(pady=10)
        
        # Frame for Input and Button
        input_frame = tk.Frame(self, bg="#90EE90")
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="Your Guess (4 digits):", bg="#90EE90").pack(side=tk.LEFT, padx=5)
        
        # Entry widget for the player's guess
        self.guess_entry = tk.Entry(input_frame, width=15)
        self.guess_entry.pack(side=tk.LEFT, padx=5)
        
        # Button to submit the guess
        self.guess_button = tk.Button(input_frame, text="Guess!", command=self.check_guess)
        self.guess_button.pack(side=tk.LEFT, padx=5)
        
        # Control Buttons
        tk.Button(self, text="New Game", command=self.start_game).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(self, text="Back to More Games", command=lambda: [self.start_game(), controller.show_frame("MoreGamesMenu")]).pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Start the first game
        self.start_game()
    
    def log(self, message):
        """Helper function to update the Text widget (Game Log)"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def generate_secret_number(self):
        """Generates a 4-digit number with unique digits."""
        digits = random.sample(range(10), 4)
        return ''.join(map(str, digits))
    
    def start_game(self):
        """Initializes the game state."""
        self.secret_number = self.generate_secret_number()
        self.attempts = 0
        
        # Clear the log and reset status
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        self.log("New game started! Guess the 4-digit number.")
        self.log("Bulls = correct digit in correct position")
        self.log("Cows = correct digit in wrong position")
        
        self.status_label.config(text=f"Attempts left: {self.max_attempts}")
        self.guess_button.config(state=tk.NORMAL)
        self.guess_entry.config(state=tk.NORMAL)
        self.guess_entry.delete(0, tk.END)
    
    def check_guess(self):
        """Handles the logic when the 'Guess!' button is pressed."""
        guess_str = self.guess_entry.get()
        self.guess_entry.delete(0, tk.END)
        
        # Validation
        if len(guess_str) != 4 or not guess_str.isdigit():
            self.log("⚠️ Please enter exactly 4 digits!")
            return
        
        if len(set(guess_str)) != 4:
            self.log("⚠️ All digits must be unique!")
            return
        
        self.attempts += 1
        
        # Calculate bulls and cows
        Bulls = sum(guess_str[i] == self.secret_number[i] for i in range(4))
        Cows = sum(guess_str[i] in self.secret_number for i in range(4)) - Bulls
        
        self.log(f"Guess #{self.attempts}: {guess_str} | Bulls: {Bulls}, Cows: {Cows}")
        
        if Bulls == 4:
            self.log(f"🎉 Correct! You guessed it in {self.attempts} attempts!")
            self.end_game()
        elif self.attempts >= self.max_attempts:
            self.log(f"❌ Game Over! The number was {self.secret_number}")
            self.end_game()
        else:
            self.status_label.config(text=f"Attempts left: {self.max_attempts - self.attempts}")
    
    def end_game(self):
        """Disables input at the end of a round."""
        self.guess_button.config(state=tk.DISABLED)
        self.guess_entry.config(state=tk.DISABLED)

# --- Word Scramble (Anagrams) GUI Frame ---
class WordScrambleGUI(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#90EE90")
        self.controller = controller
        
        # Game State Variables
        self.original_word = ""
        self.scrambled_word = ""
        self.time_left = 0
        self.timer_id = None
        
        # --- Widgets Setup ---
        tk.Label(self, text="=== Word Scramble (Anagrams) ===", font=('Arial', 16, 'bold'), bg="#90EE90").pack(pady=10)
        
        # Label for scrambled word
        self.word_label = tk.Label(self, text="", font=('Arial', 20, 'bold'), bg="#90EE90", fg="blue")
        self.word_label.pack(pady=20)
        
        # Label for timer
        self.timer_label = tk.Label(self, text="Time: 0s", font=('Arial', 14), bg="#90EE90", fg="red")
        self.timer_label.pack(pady=10)
        
        # A Text widget to act as the Game Log
        self.log_text = tk.Text(self, height=6, width=60, state=tk.DISABLED)
        self.log_text.pack(pady=10)
        
        # Frame for Input and Button
        input_frame = tk.Frame(self, bg="#90EE90")
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="Your Answer:", bg="#90EE90").pack(side=tk.LEFT, padx=5)
        
        # Entry widget for the player's answer
        self.answer_entry = tk.Entry(input_frame, width=20)
        self.answer_entry.pack(side=tk.LEFT, padx=5)
        
        # Button to submit the answer
        self.submit_button = tk.Button(input_frame, text="Submit", command=self.check_answer)
        self.submit_button.pack(side=tk.LEFT, padx=5)
        
        # Control Buttons
        tk.Button(self, text="New Game", command=self.start_game).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(self, text="Back to More Games", command=lambda: [self.stop_timer(), self.start_game(), controller.show_frame("MoreGamesMenu")]).pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Start the first game
        self.start_game()
    
    def log(self, message):
        """Helper function to update the Text widget (Game Log)"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def scramble_word(self, word):
        """Scrambles a word by randomly shuffling its letters."""
        letters = list(word)
        random.shuffle(letters)
        return ''.join(letters)
    
    def start_game(self):
        """Initializes the game state."""
        self.stop_timer()
        
        # Load a word from hangman_words
        words = load_hangman_words()
        if words == ["error"]:
            self.word_label.config(text="ERROR: Could not load words!")
            self.submit_button.config(state=tk.DISABLED)
            return
        
        self.original_word = random.choice(words).lower()
        self.scrambled_word = self.scramble_word(self.original_word)
        
        # Calculate time based on word length (3 seconds per letter, minimum 15 seconds)
        self.time_left = max(15, len(self.original_word) * 3)
        
        # Clear the log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        self.log(f"Unscramble the word! Word length: {len(self.original_word)} letters")
        
        # Update display
        self.word_label.config(text=self.scrambled_word)
        self.submit_button.config(state=tk.NORMAL)
        self.answer_entry.config(state=tk.NORMAL)
        self.answer_entry.delete(0, tk.END)
        
        # Start the timer
        self.update_timer()
    
    def update_timer(self):
        """Updates the timer display and decrements time."""
        self.timer_label.config(text=f"Time: {self.time_left}s")
        
        if self.time_left <= 0:
            self.log("⏰ Time's up! Game Over!")
            self.log(f"The word was: {self.original_word}")
            self.end_game()
        else:
            self.time_left -= 1
            self.timer_id = self.after(1000, self.update_timer)  # Update every 1 second
    
    def stop_timer(self):
        """Stops the timer."""
        if self.timer_id is not None:
            self.after_cancel(self.timer_id)
            self.timer_id = None
    
    def check_answer(self):
        """Handles the logic when the 'Submit' button is pressed."""
        answer_str = self.answer_entry.get().strip().lower()
        self.answer_entry.delete(0, tk.END)
        
        if not answer_str:
            self.log("⚠️ Please enter an answer!")
            return
        
        if answer_str == self.original_word:
            self.log(f"🎉 Correct! The word was '{self.original_word}'!")
            self.stop_timer()
            self.end_game()
        else:
            self.log(f"❌ Wrong! Try again.")
    
    def end_game(self):
        """Disables input at the end of a round."""
        self.stop_timer()
        self.submit_button.config(state=tk.DISABLED)
        self.answer_entry.config(state=tk.DISABLED)

# --- Battleship GUI Frame ---
class BattleshipGUI(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#90EE90")
        self.controller = controller
        
        # Game State Variables
        self.grid_size = 10
        self.player_boards = {
            1: [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)],
            2: [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        }
        self.player_shots = {
            1: [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)],
            2: [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        }
        
        self.player_hits = {1: 0, 2: 0}
        self.total_ship_cells = 9  # 1 3-cell + 2 2-cell + 3 1-cell = 9 cells
        self.game_phase = "MODE_SELECTION"
        self.game_mode = "COMPUTER"
        self.game_active = True
        self.current_player = 1
        self.placement_player = 1
        
        # Ship placement variables
        self.ship_sizes = [3, 2, 2, 1, 1, 1]  # Flat list of 6 ship sizes
        self.current_ship_idx = 0
        self.current_ship_size = 3
        self.placement_board_buttons = {}
        self.target_board_buttons = {}
        self.own_board_buttons = {}
        
        # --- Widgets Setup ---
        tk.Label(self, text="=== Battleship ===", font=('Arial', 16, 'bold'), bg="#90EE90").pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(self, text="Choose a game mode.", font=('Arial', 12), bg="#90EE90")
        self.status_label.pack(pady=5)
        
        # Scores (hidden during placement)
        self.score_label = tk.Label(self, text="", font=('Arial', 11), bg="#90EE90")
        
        # Placement instructions (shown during placement)
        self.instruction_label = tk.Label(self, text="Ship placement goes here", font=('Arial', 10), bg="#90EE90")
        
        # Main content frame
        self.content_frame = tk.Frame(self, bg="#90EE90")
        self.content_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Game log
        self.log_text = tk.Text(self, height=6, width=40, state=tk.DISABLED)
        self.log_text.pack(pady=5)
        
        # Frame for Input
        self.input_frame = tk.Frame(self, bg="#90EE90")
        
        # Letter entry
        self.letter_entry = tk.Entry(self.input_frame, width=3)
        
        # Number entry
        self.number_entry = tk.Entry(self.input_frame, width=3)
        
        # Fire button
        self.fire_button = tk.Button(self.input_frame, text="Fire!", command=self.player_fire)
        
        # Control Buttons frame
        self.button_frame = tk.Frame(self, bg="#90EE90")
        
        self.show_mode_selection()
    
    def show_mode_selection(self):
        """Show a simple mode selection screen before the game begins."""
        self.game_phase = "MODE_SELECTION"
        self.game_mode = "COMPUTER"
        self.game_active = True
        self.current_player = 1
        self.placement_player = 1
        self.status_label.config(text="Choose a Battleship mode")
        self.score_label.pack_forget()
        self.instruction_label.pack_forget()
        self.input_frame.pack_forget()
        self.button_frame.pack_forget()
        
        self.content_frame.pack_forget()
        self.content_frame.destroy()
        self.content_frame = tk.Frame(self, bg="#90EE90")
        self.content_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        mode_frame = tk.Frame(self.content_frame, bg="#90EE90")
        mode_frame.pack(expand=True)
        
        tk.Label(mode_frame, text="Select Battleship Mode", font=('Arial', 14, 'bold'), bg="#90EE90").pack(pady=10)
        tk.Label(mode_frame, text="Play against the computer or a second player", font=('Arial', 11), bg="#90EE90").pack(pady=5)
        tk.Button(mode_frame, text="1 Player (vs Computer)", width=22, command=lambda: self.start_mode("COMPUTER")).pack(pady=8)
        tk.Button(mode_frame, text="2 Players (Hot Seat)", width=22, command=lambda: self.start_mode("TWOPLAYER")).pack(pady=8)
    
    def start_mode(self, mode):
        """Start a new game in the selected mode."""
        self.game_mode = mode
        self.game_active = True
        self.current_player = 1
        self.placement_player = 1
        self.player_hits = {1: 0, 2: 0}
        self.player_boards = {
            1: [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)],
            2: [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        }
        self.player_shots = {
            1: [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)],
            2: [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        }
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.init_placement_phase()
    
    def init_placement_phase(self):
        """Initialize the ship placement phase."""
        self.game_phase = "PLACEMENT"
        self.current_ship_idx = 0
        self.current_ship_size = self.ship_sizes[0]
        
        if self.game_mode == "TWOPLAYER":
            if self.placement_player == 1:
                self.status_label.config(text="Player 1, place your ships.")
            else:
                self.status_label.config(text="Player 2, place your ships.")
        else:
            self.status_label.config(text="Place your ships! Click on the board.")
        
        self.instruction_label.config(text=f"Placing ship of size {self.current_ship_size} (ship 1 of 6)")
        self.instruction_label.pack(pady=5)
        self.score_label.pack_forget()
        
        # Clear content frame efficiently
        self.content_frame.pack_forget()
        self.content_frame.destroy()
        self.content_frame = tk.Frame(self, bg="#90EE90")
        self.content_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Create placement board
        placement_bg = "#E6F7FF" if self.placement_player == 1 else "#FFE6E6"
        cell_bg = "lightblue" if self.placement_player == 1 else "lightpink"
        board_frame = tk.Frame(self.content_frame, bg=placement_bg, relief=tk.SUNKEN, bd=2)
        board_frame.pack(padx=10, fill=tk.BOTH, expand=True)
        
        if self.game_mode == "TWOPLAYER":
            board_title = f"Player {self.placement_player} - Place Ships"
        else:
            board_title = "Your Board - Place Ships"
        tk.Label(board_frame, text=board_title, font=('Arial', 10, 'bold'), bg=placement_bg).pack()
        
        grid_frame = tk.Frame(board_frame, bg=placement_bg)
        grid_frame.pack(padx=5, pady=5)
        
        tk.Label(grid_frame, text="  ", bg=placement_bg, width=2).grid(row=0, column=0)
        for col in range(self.grid_size):
            tk.Label(grid_frame, text=chr(ord('A') + col), bg=placement_bg, width=2, font=('Arial', 8, 'bold')).grid(row=0, column=col + 1)
        
        self.placement_board_buttons = {}
        for row in range(self.grid_size):
            tk.Label(grid_frame, text=str(row + 1), bg=placement_bg, width=2, font=('Arial', 8, 'bold')).grid(row=row + 1, column=0)
            for col in range(self.grid_size):
                btn = tk.Label(grid_frame, text=" ", bg=cell_bg, width=2, height=1, font=('Arial', 8), relief=tk.RAISED, bd=1)
                btn.grid(row=row + 1, column=col + 1)
                btn.bind("<Button-1>", lambda e, r=row, c=col: self.place_ship_on_board(r, c))
                self.placement_board_buttons[(row, col)] = btn
        
        # Placement buttons
        placement_btn_frame = tk.Frame(self.content_frame, bg="#90EE90")
        placement_btn_frame.pack(pady=10)
        tk.Button(placement_btn_frame, text="Random Placement", command=self.random_placement).pack(side=tk.LEFT, padx=5)
        if self.game_mode == "TWOPLAYER" and self.placement_player == 1:
            tk.Button(placement_btn_frame, text="Second Player", command=self.start_game_from_placement).pack(side=tk.LEFT, padx=5)
        else:
            tk.Button(placement_btn_frame, text="Start Game", command=self.start_game_from_placement).pack(side=tk.LEFT, padx=5)
    
    def place_ship_on_board(self, row, col):
        """Try to place a ship at the given coordinates."""
        board = self.player_boards[self.placement_player if self.game_mode == "TWOPLAYER" else 1]
        if not self.can_place_ship(board, row, col, self.current_ship_size, 'H'):
            if not self.can_place_ship(board, row, col, self.current_ship_size, 'V'):
                self.instruction_label.config(text="Can't place there! Try another spot.")
                return
            direction = 'V'
        else:
            direction = 'H'
        
        # Place the ship
        for i in range(self.current_ship_size):
            if direction == 'H':
                board[row][col + i] = 'S'
                self.placement_board_buttons[(row, col + i)].config(bg="gray")
            else:
                board[row + i][col] = 'S'
                self.placement_board_buttons[(row + i, col)].config(bg="gray")
        
        # Move to next ship
        self.current_ship_idx += 1
        if self.current_ship_idx >= 6:  # 6 total ships
            if self.game_mode == "TWOPLAYER" and self.placement_player == 1:
                self.instruction_label.config(text="Player 1 ships are ready. Click 'Second Player' to continue.")
            elif self.game_mode == "TWOPLAYER" and self.placement_player == 2:
                self.instruction_label.config(text="Player 2 ships are ready. Click 'Start Game'.")
            else:
                self.instruction_label.config(text="All ships placed! Click 'Start Game'.")
        else:
            self.current_ship_size = self.ship_sizes[self.current_ship_idx]
            ship_num = self.current_ship_idx + 1
            self.instruction_label.config(text=f"Placing ship of size {self.current_ship_size} (ship {ship_num} of 6)")
    
    def random_placement(self):
        """Randomly place all remaining ships."""
        board = self.player_boards[self.placement_player if self.game_mode == "TWOPLAYER" else 1]
        board[:] = [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.place_ships(board)
        
        # Update display using the active player's placement colors
        cell_bg = "lightblue" if self.placement_player == 1 else "lightpink"
        for (row, col), btn in self.placement_board_buttons.items():
            if board[row][col] == 'S':
                btn.config(bg="gray")
            else:
                btn.config(bg=cell_bg)
        
        self.instruction_label.config(text="Ships randomly placed! Click 'Start Game'")
        self.current_ship_idx = 6
    
    def start_game_from_placement(self):
        """Transition from placement phase to game phase."""
        if self.current_ship_idx < 6:
            self.instruction_label.config(text="Place all ships first!")
            return
        
        if self.game_mode == "TWOPLAYER" and self.placement_player == 1:
            self.placement_player = 2
            self.current_ship_idx = 0
            self.current_ship_size = self.ship_sizes[0]
            self.init_placement_phase()
            return
        
        self.game_phase = "PLAYING"
        if self.game_mode == "COMPUTER":
            self.place_ships(self.player_boards[2])
        self.instruction_label.config(text="")
        self.instruction_label.pack_forget()
        self.score_label.pack(pady=5)
        self.init_game_phase()
    
    def init_game_phase(self):
        """Initialize the actual game phase with two boards."""
        # Clear content frame efficiently
        self.content_frame.pack_forget()
        self.content_frame.destroy()
        self.content_frame = tk.Frame(self, bg="#90EE90")
        self.content_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Main content frame with two boards
        boards_frame = tk.Frame(self.content_frame, bg="#90EE90")
        boards_frame.pack(padx=10, fill=tk.BOTH, expand=True)
        
        if self.game_mode == "COMPUTER":
            left_title = "Computer's Board"
            right_title = "Your Board"
        else:
            left_title = "Player 1's Board"
            right_title = "Player 2's Board"
        
        # LEFT BOARD: Opponent's Board (where player fires)
        left_frame = tk.Frame(boards_frame, bg="white", relief=tk.SUNKEN, bd=2)
        left_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        
        tk.Label(left_frame, text=left_title, font=('Arial', 10, 'bold'), bg="white").pack()
        
        left_grid = tk.Frame(left_frame, bg="white")
        left_grid.pack(padx=5, pady=5)
        
        tk.Label(left_grid, text="  ", bg="white", width=2).grid(row=0, column=0)
        for col in range(self.grid_size):
            tk.Label(left_grid, text=chr(ord('A') + col), bg="white", width=2, font=('Arial', 8, 'bold')).grid(row=0, column=col + 1)
        
        self.target_board_buttons = {}
        for row in range(self.grid_size):
            tk.Label(left_grid, text=str(row + 1), bg="white", width=2, font=('Arial', 8, 'bold')).grid(row=row + 1, column=0)
            for col in range(self.grid_size):
                btn = tk.Label(left_grid, text=" ", bg="lightblue", width=2, height=1, font=('Arial', 8), relief=tk.RAISED, bd=1)
                btn.grid(row=row + 1, column=col + 1)
                self.target_board_buttons[(row, col)] = btn
        
        # RIGHT BOARD: Current player's board (where opponent fires)
        right_frame = tk.Frame(boards_frame, bg="white", relief=tk.SUNKEN, bd=2)
        right_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        
        tk.Label(right_frame, text=right_title, font=('Arial', 10, 'bold'), bg="white").pack()
        
        right_grid = tk.Frame(right_frame, bg="white")
        right_grid.pack(padx=5, pady=5)
        
        tk.Label(right_grid, text="  ", bg="white", width=2).grid(row=0, column=0)
        for col in range(self.grid_size):
            tk.Label(right_grid, text=chr(ord('A') + col), bg="white", width=2, font=('Arial', 8, 'bold')).grid(row=0, column=col + 1)
        
        self.own_board_buttons = {}
        for row in range(self.grid_size):
            tk.Label(right_grid, text=str(row + 1), bg="white", width=2, font=('Arial', 8, 'bold')).grid(row=row + 1, column=0)
            for col in range(self.grid_size):
                btn = tk.Label(right_grid, text=" ", bg="lightgreen", width=2, height=1, font=('Arial', 8), relief=tk.RAISED, bd=1)
                btn.grid(row=row + 1, column=col + 1)
                self.own_board_buttons[(row, col)] = btn
        
        # Log and input frame
        log_input_frame = tk.Frame(self.content_frame, bg="#90EE90")
        log_input_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH)
        
        self.log_text.config(height=8, width=30, state=tk.DISABLED)
        self.log_text.pack(pady=5)
        
        # Input frame
        self.input_frame.destroy()
        self.input_frame = tk.Frame(log_input_frame, bg="#90EE90")
        self.input_frame.pack(pady=5)
        
        tk.Label(self.input_frame, text="Fire at:", bg="#90EE90").pack(side=tk.LEFT, padx=5)
        tk.Label(self.input_frame, text="Letter:", bg="#90EE90").pack(side=tk.LEFT, padx=2)
        self.letter_entry = tk.Entry(self.input_frame, width=3)
        self.letter_entry.pack(side=tk.LEFT, padx=2)
        tk.Label(self.input_frame, text="Number:", bg="#90EE90").pack(side=tk.LEFT, padx=2)
        self.number_entry = tk.Entry(self.input_frame, width=3)
        self.number_entry.pack(side=tk.LEFT, padx=2)
        self.fire_button = tk.Button(self.input_frame, text="Fire!", command=self.player_fire)
        self.fire_button.pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        self.button_frame.destroy()
        self.button_frame = tk.Frame(self, bg="#90EE90")
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        tk.Button(self.button_frame, text="New Game", command=self.reset_to_placement).pack(side=tk.LEFT, padx=10)
        tk.Button(self.button_frame, text="Back to More Games", command=self.back_to_menu).pack(side=tk.RIGHT, padx=10)
        
        self.log(f"Game started! {'Computer' if self.game_mode == 'COMPUTER' else f'Player {self.current_player}'} to move.")
        self.update_all_boards()
        self.update_score_display()
    
    def update_all_boards(self):
        """Updates the fixed boards for both players without swapping them on each turn."""
        if self.game_mode == "COMPUTER":
            # Keep the single-player view unchanged.
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    btn = self.target_board_buttons[(row, col)]
                    shot = self.player_shots[2][row][col]
                    if shot == 'H':
                        btn.config(bg="red", text="H")
                    elif shot == 'M':
                        btn.config(bg="lightgray", text="M")
                    else:
                        btn.config(bg="lightblue", text=" ")
            
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    btn = self.own_board_buttons[(row, col)]
                    shot = self.player_shots[1][row][col]
                    if self.player_boards[1][row][col] == 'S':
                        if shot == 'H':
                            btn.config(bg="darkred", text="H")
                        elif shot == 'M':
                            btn.config(bg="lightgreen", text=" ")
                        else:
                            btn.config(bg="lightgreen", text=" ")
                    else:
                        if shot == 'M':
                            btn.config(bg="lightgray", text="M")
                        else:
                            btn.config(bg="lightgreen", text=" ")
            return

        # Two-player mode: keep Player 1's board and Player 2's board fixed.
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                btn = self.target_board_buttons[(row, col)]
                shot = self.player_shots[2][row][col]
                if shot == 'H':
                    btn.config(bg="red", text="H")
                elif shot == 'M':
                    btn.config(bg="lightgray", text="M")
                else:
                    btn.config(bg="lightblue", text=" ")

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                btn = self.own_board_buttons[(row, col)]
                shot = self.player_shots[1][row][col]
                if self.player_boards[2][row][col] == 'S':
                    if shot == 'H':
                        btn.config(bg="darkred", text="H")
                    elif shot == 'M':
                        btn.config(bg="lightgreen", text=" ")
                    else:
                        btn.config(bg="lightgreen", text=" ")
                else:
                    if shot == 'M':
                        btn.config(bg="lightgray", text="M")
                    else:
                        btn.config(bg="lightgreen", text=" ")
    
    def log(self, message):
        """Helper function to update the Text widget (Game Log)"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def place_ships(self, board):
        """Places ships on the board randomly."""
        # Ship configurations: [size, count]
        ships = [(3, 1), (2, 2), (1, 3)]  # 1 3-length, 2 2-length, 3 1-length
        
        for size, count in ships:
            for _ in range(count):
                placed = False
                while not placed:
                    row = random.randint(0, self.grid_size - 1)
                    col = random.randint(0, self.grid_size - 1)
                    direction = random.choice(['H', 'V'])  # Horizontal or Vertical
                    
                    # Check if placement is valid
                    if self.can_place_ship(board, row, col, size, direction):
                        # Place the ship
                        for i in range(size):
                            if direction == 'H':
                                board[row][col + i] = 'S'  # S for ship
                            else:
                                board[row + i][col] = 'S'
                        placed = True
    
    def can_place_ship(self, board, row, col, size, direction):
        """Checks if a ship can be placed at the given position."""
        if direction == 'H':
            if col + size > self.grid_size:
                return False
            for i in range(size):
                if board[row][col + i] == 'S':
                    return False
        else:  # Vertical
            if row + size > self.grid_size:
                return False
            for i in range(size):
                if board[row + i][col] == 'S':
                    return False
        return True
    
    
    def reset_to_placement(self):
        """Go back to the placement phase."""
        self.show_mode_selection()
    
    def back_to_menu(self):
        """Return to more games menu and reset window size."""
        self.show_mode_selection()
        self.controller.winfo_toplevel().geometry("600x550")
        self.controller.show_frame("MoreGamesMenu")
    
    def update_score_display(self):
        """Updates the score display."""
        if self.game_mode == "COMPUTER":
            self.score_label.config(text=f"Your Hits: {self.player_hits[1]}/{self.total_ship_cells} | Computer Hits: {self.player_hits[2]}/{self.total_ship_cells}")
        else:
            self.score_label.config(text=f"Player 1 Hits: {self.player_hits[1]}/{self.total_ship_cells} | Player 2 Hits: {self.player_hits[2]}/{self.total_ship_cells}")
    
    def coordinates_to_index(self, letter, number):
        """Converts letter and number to row and column indices."""
        try:
            col = ord(letter.upper()) - ord('A')
            row = int(number) - 1
            
            if col < 0 or col >= self.grid_size or row < 0 or row >= self.grid_size:
                return None, None
            
            return row, col
        except (ValueError, TypeError):
            return None, None
    
    def index_to_coordinates(self, row, col):
        """Converts row and column indices to letter and number."""
        letter = chr(ord('A') + col)
        number = row + 1
        return f"{letter} {number}"
    
    def player_name(self, player_num):
        """Return a friendly name for each player."""
        if self.game_mode == "COMPUTER" and player_num == 2:
            return "Computer"
        return f"Player {player_num}"
    
    def player_fire(self):
        """Handles the current player's fire action."""
        letter = self.letter_entry.get().strip()
        number = self.number_entry.get().strip()
        
        self.letter_entry.delete(0, tk.END)
        self.number_entry.delete(0, tk.END)
        
        row, col = self.coordinates_to_index(letter, number)
        
        if row is None or col is None:
            self.log("⚠️ Invalid coordinates! Use format: A 5")
            return
        
        current_player = self.current_player
        target_player = 2 if current_player == 1 else 1
        if self.game_mode == "COMPUTER":
            target_player = 2
        
        if self.player_shots[current_player][row][col] != ' ':
            self.log("⚠️ You already fired at that location!")
            return
        
        # Fire at the target player's board
        if self.player_boards[target_player][row][col] == 'S':
            self.player_shots[current_player][row][col] = 'H'  # Hit
            self.player_hits[current_player] += 1
            coords = self.index_to_coordinates(row, col)
            self.log(f"🎯 {self.player_name(current_player)} hit at {coords}!")
            
            if self.player_hits[current_player] >= self.total_ship_cells:
                self.log(f"🎉 {self.player_name(current_player)} sank all ships! {self.player_name(current_player)} wins!")
                self.game_active = False
                self.end_game()
                return
        else:
            self.player_shots[current_player][row][col] = 'M'  # Miss
            coords = self.index_to_coordinates(row, col)
            self.log(f"❌ {self.player_name(current_player)} missed at {coords}.")
        
        self.update_all_boards()
        self.update_score_display()
        
        if self.game_mode == "COMPUTER":
            self.computer_fire()
        else:
            self.switch_turn()
    
    def switch_turn(self):
        """Switch to the next player in two-player mode."""
        if not self.game_active:
            return
        self.current_player = 2 if self.current_player == 1 else 1
        self.log(f"🔄 {self.player_name(self.current_player)}'s turn")
        self.status_label.config(text=f"{self.player_name(self.current_player)}'s turn")
        self.update_all_boards()
        self.update_score_display()
    
    def computer_fire(self):
        """Handles computer's fire action (random for now)."""
        row = random.randint(0, self.grid_size - 1)
        col = random.randint(0, self.grid_size - 1)
        
        # Find an unfired location
        attempts = 0
        while self.player_shots[2][row][col] != ' ' and attempts < 100:
            row = random.randint(0, self.grid_size - 1)
            col = random.randint(0, self.grid_size - 1)
            attempts += 1
        
        if self.player_shots[2][row][col] != ' ':
            return  # No valid moves left (shouldn't happen)
        
        # Fire at player's board
        if self.player_boards[1][row][col] == 'S':
            self.player_shots[2][row][col] = 'H'  # Hit
            self.player_hits[2] += 1
            coords = self.index_to_coordinates(row, col)
            self.log(f"💥 Computer hit at {coords}!")
            
            if self.player_hits[2] >= self.total_ship_cells:
                self.log("💻 Computer sank all your ships! You lose!")
                self.game_active = False
                self.end_game()
                return
        else:
            self.player_shots[2][row][col] = 'M'  # Miss
            coords = self.index_to_coordinates(row, col)
            self.log(f"💭 Computer fired at {coords} and missed.")
        
        self.update_all_boards()
        self.update_score_display()
    
    def end_game(self):
        """Disables input at the end of the game."""
        self.fire_button.config(state=tk.DISABLED)
        self.letter_entry.config(state=tk.DISABLED)
        self.number_entry.config(state=tk.DISABLED)

# --- Slot Machine Game GUI Frame ---
class SlotMachineGUI(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#90EE90")
        self.controller = controller
        
        # Game State Variables
        self.balance = 100
        self.symbols = ["🍒", "🍋", "🔔", "💎"]
        self.current_reels = ["🍒", "🍋", "🔔"]
        self.spinning = False
        
        # --- Widgets Setup ---
        tk.Label(self, text="=== Slot Machine ===", font=('Arial', 16, 'bold'), bg="#90EE90").pack(pady=10)
        
        # Balance display
        self.balance_label = tk.Label(self, text=f"💰 Balance: ${self.balance}", font=('Arial', 14, 'bold'), bg="#90EE90")
        self.balance_label.pack(pady=10)
        
        # Reel display (three large symbols)
        reel_frame = tk.Frame(self, bg="#90EE90")
        reel_frame.pack(pady=20)
        
        self.reel_labels = []
        for i in range(3):
            label = tk.Label(reel_frame, text=self.current_reels[i], font=('Arial', 60, 'bold'), 
                            bg="white", width=4, relief=tk.SUNKEN, bd=3)
            label.pack(side=tk.LEFT, padx=10)
            self.reel_labels.append(label)
        
        # Bet input frame
        bet_frame = tk.Frame(self, bg="#90EE90")
        bet_frame.pack(pady=15)
        
        tk.Label(bet_frame, text="Bet Amount:", font=('Arial', 12), bg="#90EE90").pack(side=tk.LEFT, padx=5)
        self.bet_entry = tk.Entry(bet_frame, width=10, font=('Arial', 12))
        self.bet_entry.pack(side=tk.LEFT, padx=5)
        self.bet_entry.insert(0, "10")  # Default bet
        
        # Quick bet buttons
        quick_bet_frame = tk.Frame(self, bg="#90EE90")
        quick_bet_frame.pack(pady=5)
        
        for amount in [5, 10, 20, 50]:
            tk.Button(quick_bet_frame, text=f"${amount}", width=6, 
                     command=lambda a=amount: self.set_bet(a), bg="lightblue").pack(side=tk.LEFT, padx=3)
        
        # Spin button
        self.spin_button = tk.Button(self, text="SPIN!", font=('Arial', 14, 'bold'), 
                                     bg="gold", width=20, command=self.spin)
        self.spin_button.pack(pady=15)
        
        # Result message
        self.result_label = tk.Label(self, text="Place a bet and spin!", font=('Arial', 12), bg="#90EE90", fg="blue")
        self.result_label.pack(pady=10)
        
        # Control buttons
        button_frame = tk.Frame(self, bg="#90EE90")
        button_frame.pack(pady=15)
        
        tk.Button(button_frame, text="New Game (Reset)", command=self.reset_game, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Back to Menu", 
                 command=lambda: [self.reset_game(), controller.show_frame("MoreGamesMenu")], width=15).pack(side=tk.LEFT, padx=5)
    
    def set_bet(self, amount):
        """Set the bet amount from quick buttons."""
        self.bet_entry.delete(0, tk.END)
        self.bet_entry.insert(0, str(amount))
    
    def spin(self):
        """Spin the reels."""
        if self.spinning:
            return
        
        # Get bet amount
        try:
            bet = int(self.bet_entry.get())
            if bet <= 0:
                self.result_label.config(text="Bet must be positive!", fg="red")
                return
            if bet > self.balance:
                self.result_label.config(text="Not enough balance!", fg="red")
                return
        except ValueError:
            self.result_label.config(text="Invalid bet amount!", fg="red")
            return
        
        self.spinning = True
        self.spin_button.config(state=tk.DISABLED)
        
        # Spin animation
        self.animate_spin(bet, 0)
    
    def animate_spin(self, bet, frame):
        """Animate spinning reels."""
        if frame < 15:  # Spin for 15 frames
            for i in range(3):
                self.current_reels[i] = random.choice(self.symbols)
                self.reel_labels[i].config(text=self.current_reels[i])
            self.after(100, lambda: self.animate_spin(bet, frame + 1))
        else:
            # Spin complete, calculate result
            self.calculate_result(bet)
            self.spinning = False
            self.spin_button.config(state=tk.NORMAL)
    
    def calculate_result(self, bet):
        """Calculate win/loss and update balance."""
        reel1, reel2, reel3 = self.current_reels
        
        # Check for matches
        matches = sum([reel1 == reel2, reel2 == reel3, reel1 == reel3])
        
        winnings = 0
        if reel1 == reel2 == reel3:  # All three match
            winnings = bet * 10
            message = f"🎉 THREE OF A KIND! Won ${winnings}!"
            color = "green"
        elif matches == 1:  # Two match
            winnings = bet * 3
            message = f"✨ TWO MATCH! Won ${winnings}!"
            color = "green"
        else:  # No match
            message = f"💔 No match. Lost ${bet}."
            color = "red"
        
        self.balance -= bet
        self.balance += winnings
        
        # Update displays
        self.balance_label.config(text=f"💰 Balance: ${self.balance}")
        self.result_label.config(text=message, fg=color)
        
        # Check if player is out of money
        if self.balance <= 0:
            self.result_label.config(text="Game Over! You're out of money!", fg="red")
            self.spin_button.config(state=tk.DISABLED)
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.balance = 100
        self.current_reels = ["🍒", "🍋", "🔔"]
        self.spinning = False
        
        self.balance_label.config(text=f"💰 Balance: ${self.balance}")
        self.result_label.config(text="Place a bet and spin!", fg="blue")
        self.spin_button.config(state=tk.NORMAL)
        self.bet_entry.delete(0, tk.END)
        self.bet_entry.insert(0, "10")
        
        for i, label in enumerate(self.reel_labels):
            label.config(text=self.current_reels[i])

# --- Text Adventure (Escape Room) GUI Frame ---
class TextAdventureGUI(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#2C3E50")
        self.controller = controller
        
        # Game State Variables
        self.current_scene = "main_room"
        self.has_silver_key = False
        self.has_iron_key = False
        self.has_seen_code = False  # Code only appears after visiting desk
        self.in_code_entry = False  # Flag to prevent recursion
        
        # Define all scenes and choices
        self.scenes = {
            "main_room": {
                "title": "The Locked Study",
                "description": "You wake up with a headache in a dimly lit, dusty study. You have no memory of how you got here. "
                               "The only light comes from a flickering candle. The air is stale. You need to get out. "
                               "You look around and see three things of interest: a heavy oak Door, a cluttered Desk, and a tall Bookshelf.\n\n"
                               "What do you want to inspect?",
                "choices": [
                    ("1. Inspect the Door", "door_inspect"),
                    ("2. Inspect the Desk", "desk_inspect"),
                    ("3. Inspect the Bookshelf", "bookshelf_inspect"),
                ]
            },
            "door_inspect": {
                "title": "The Heavy Oak Door",
                "description": None,  # Will be set dynamically
                "choices": [
                    ("1. Go back", "main_room"),
                ]
            },
            "door_escape": {
                "title": "🎉 FREEDOM!",
                "description": "You slide the heavy Iron Key into the lock. It turns with a satisfying CLICK. "
                               "You push the door open and cool, fresh air hits your face. You have escaped!",
                "choices": [
                    ("1. Play Again", "restart"),
                ]
            },
            "desk_inspect": {
                "title": "The Cluttered Desk",
                "description": "You walk over to the desk. It is covered in dust. You find a torn piece of paper with the numbers "
                               "'7-3-8-4' scribbled on it in red ink. You also notice a small drawer, but it is locked with a tiny silver padlock.",
                "choices": [
                    ("1. Try to force the drawer open", "desk_force_drawer"),
                    ("2. Leave the desk", "main_room"),
                ]
            },
            "desk_force_drawer": {
                "title": "The Desk Drawer",
                "description": None,  # Will be set dynamically
                "choices": [
                    ("1. Go back to the desk", "desk_inspect"),
                    ("1a. Return to main room", "main_room"),
                ]
            },
            "bookshelf_inspect": {
                "title": "The Tall Bookshelf",
                "description": "The bookshelf is filled with thick, ancient books. You notice one book looks completely different—"
                               "it has no title and is made of metal. You pull it out and realize it's actually a lockbox! "
                               "It requires a 4-digit passcode.",
                "choices": [
                    ("1. Enter a 4-digit code", "bookshelf_code_entry"),
                    ("2. Put the box down and walk away", "main_room"),
                ]
            },
            "bookshelf_code_entry": {
                "title": "Metal Lockbox - Code Entry",
                "description": "Enter the 4-digit code from the desk to open the lockbox...",  # Will be updated after code entry
                "choices": [],  # Will be set dynamically
            },
            "bookshelf_code_correct": {
                "title": "The Metal Lockbox Opens!",
                "description": "A green light flashes, and the box pops open! Inside, resting on a velvet cushion, is a small Silver Key.",
                "choices": [
                    ("1. Take the Silver Key and return to main room", "main_room"),
                ]
            },
            "bookshelf_already_open": {
                "title": "The Bookshelf",
                "description": "You look at the bookshelf. The metal lockbox is already open and empty. There is nothing else of interest here.",
                "choices": [
                    ("1. Return to main room", "main_room"),
                ]
            },
            "restart": {
                "title": "New Game",
                "description": "Restarting...",
                "choices": [
                    ("1. Begin", "main_room"),
                ]
            }
        }
        
        # --- Widgets Setup ---
        tk.Label(self, text="=== The Locked Study: Escape Room ===", font=('Arial', 16, 'bold'), 
                bg="#2C3E50", fg="#ECF0F1").pack(pady=10)
        
        # Scene title
        self.title_label = tk.Label(self, text="", font=('Arial', 14, 'bold'), 
                                   bg="#2C3E50", fg="#3498DB", wraplength=500)
        self.title_label.pack(pady=10)
        
        # Scene description
        self.description_text = tk.Text(self, height=12, width=80, state=tk.DISABLED, 
                                        bg="#34495E", fg="#ECF0F1", font=('Arial', 11), wrap=tk.WORD)
        self.description_text.pack(pady=15, padx=20, fill=tk.BOTH, expand=True)
        
        # Inventory display
        self.inventory_label = tk.Label(self, text="Inventory: Code (7-3-8-4)", font=('Arial', 10, 'bold'), 
                                       bg="#2C3E50", fg="#F39C12")
        self.inventory_label.pack(pady=5)
        
        # Choices frame
        self.choices_frame = tk.Frame(self, bg="#2C3E50")
        self.choices_frame.pack(pady=15, fill=tk.BOTH, expand=True)
        
        # Control Buttons
        button_frame = tk.Frame(self, bg="#2C3E50")
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Restart Game", command=self.restart_game, 
                 bg="#E74C3C", fg="white", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Back to Menu", 
                 command=lambda: [self.restart_game(), controller.winfo_toplevel().geometry("600x500"), controller.show_frame("MoreGamesMenu")],
                 bg="#27AE60", fg="white", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Start the game
        self.show_scene("main_room")
    
    def show_scene(self, scene_key):
        """Display a scene and its choices."""
        if scene_key == "restart":
            self.restart_game()
            return
        
        if scene_key not in self.scenes:
            scene_key = "main_room"
        
        self.current_scene = scene_key
        scene = self.scenes[scene_key]
        
        # Dynamic scene content based on inventory
        if scene_key == "door_inspect":
            if self.has_iron_key:
                self.show_scene("door_escape")
                return
            else:
                description = "You pull on the brass handle, but it doesn't budge. There is a heavy, old-fashioned keyhole. " \
                             "You need to find the right key to open this door."
                scene["description"] = description
        
        elif scene_key == "desk_inspect":
            self.has_seen_code = True  # Player finds the code here
        
        elif scene_key == "desk_force_drawer":
            if self.has_silver_key:
                description = "You use the small Silver Key on the tiny padlock. It pops open! Inside the drawer, " \
                             "you find a heavy, rusty Iron Key."
                self.has_iron_key = True
                scene["description"] = description
            else:
                description = "You pull hard, but the wood is too sturdy. You need a key."
                scene["description"] = description
        
        elif scene_key == "bookshelf_inspect":
            if self.has_silver_key:
                self.show_scene("bookshelf_already_open")
                return
        
        elif scene_key == "bookshelf_code_entry" and not self.in_code_entry:
            self.in_code_entry = True
            self.prompt_code_entry()
            self.in_code_entry = False
            return
        
        # Update title
        self.title_label.config(text=scene["title"])
        
        # Update description
        self.description_text.config(state=tk.NORMAL)
        self.description_text.delete(1.0, tk.END)
        if scene["description"] is not None:
            self.description_text.insert(tk.END, scene["description"])
        self.description_text.config(state=tk.DISABLED)
        
        # Update inventory display
        inv_items = []
        if self.has_seen_code:
            inv_items.append("Code (7-3-8-4)")
        if self.has_silver_key:
            inv_items.append("Silver Key")
        if self.has_iron_key:
            inv_items.append("Iron Key")
        
        if inv_items:
            self.inventory_label.config(text="Inventory: " + ", ".join(inv_items))
        else:
            self.inventory_label.config(text="Inventory: Empty")
        
        # Clear previous choices
        for widget in self.choices_frame.winfo_children():
            widget.destroy()
        
        # Display choice buttons
        if scene["choices"]:
            for choice_text, next_scene in scene["choices"]:
                btn = tk.Button(self.choices_frame, text=choice_text, width=60, 
                               command=lambda s=next_scene: self.show_scene(s),
                               bg="#3498DB", fg="white", font=('Arial', 11),
                               relief=tk.RAISED, bd=2, activebackground="#2980B9")
                btn.pack(pady=5)
    
    def prompt_code_entry(self):
        """Prompt player to enter the 4-digit code."""
        result = simpledialog.askstring("Enter Code", "Enter the 4-digit code from the desk:", parent=self)
        
        if result is None:  # User cancelled
            self.show_scene("bookshelf_inspect")
            return
        
        result = result.strip()
        
        if result == "7384" or result == "7-3-8-4":
            self.has_silver_key = True
            self.scenes["bookshelf_code_entry"]["description"] = \
                "A green light flashes, and the box pops open! Inside, resting on a velvet cushion, is a small Silver Key."
            self.scenes["bookshelf_code_entry"]["choices"] = \
                [("1. Take the Silver Key and return to main room", "main_room")]
        else:
            self.scenes["bookshelf_code_entry"]["description"] = \
                "The box beeps angrily and flashes a red light. The code is incorrect. " \
                "You hear the lockbox reset with a mechanical click."
            self.scenes["bookshelf_code_entry"]["choices"] = \
                [("1. Try again", "bookshelf_code_entry"), ("2. Give up and return", "main_room")]
        
        self.show_scene("bookshelf_code_entry")
    
    def restart_game(self):
        """Reset the game to the start."""
        self.current_scene = "main_room"
        self.has_silver_key = False
        self.has_iron_key = False
        self.has_seen_code = False
        self.in_code_entry = False
        self.show_scene("main_room")

# --- Game WIP Screen ---
class GameWIP(tk.Frame):
    """Placeholder screen for work in progress games."""
    def __init__(self, parent, controller):
        bg="#90EE90"
        tk.Frame.__init__(self, parent, bg=bg) 
        self.controller = controller

        tk.Label(self, text="=== Game Coming Soon ===", font=('Arial', 16, 'bold'), bg=bg).pack(pady=20)
        
        tk.Label(self, text="I Work!", font=('Arial', 24, 'bold'), bg=bg).pack(pady=50)

        tk.Button(self, text="Back to More Games", width=30, command=lambda: controller.show_frame("MoreGamesMenu")).pack(pady=20)

# --- Number Guessing Game GUI Frame ---
class NumberGuessingGUI(tk.Frame):
    def __init__(self, parent, controller):
        bg="#90EE90"
        tk.Frame.__init__(self, parent, bg=bg)
        self.controller = controller
        
        self.max_attempts = 10
        self.target_number = 0
        self.attempts_left = 0

        # --- Widgets Setup ---
        tk.Label(self, text="=== Number Guessing Game ===", font=('Arial', 14, 'bold'), bg=bg).pack(pady=10)
        
        # A Text widget to act as the Game Log
        self.log_text = tk.Text(self, height=8, width=60, state=tk.DISABLED)
        self.log_text.pack(pady=10)

        # Frame for Input and Button
        input_frame = tk.Frame(self, bg=bg)
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="Your Guess (1-10):", bg=bg).pack(side=tk.LEFT, padx=5)
        
        # Entry widget for the player's guess
        self.guess_entry = tk.Entry(input_frame, width=10)
        self.guess_entry.pack(side=tk.LEFT, padx=5)
        
        # Button to submit the guess
        self.guess_button = tk.Button(input_frame, text="Guess!", command=self.check_guess)
        self.guess_button.pack(side=tk.LEFT, padx=5)

        # Label to show status (Attempts left)
        self.status_label = tk.Label(self, text="", bg=bg)
        self.status_label.pack(pady=10)

        # Button to start a new game
        tk.Button(self, text="New Game", command=self.start_game).pack(pady=5)
        
        # Button to return to the main menu
        tk.Button(self, text="Back to Menu", command=lambda: [self.start_game(), controller.show_frame("MainMenu")]).pack(pady=5)

        # Start the first game automatically
        self.start_game()
        
    def log(self, message):
        """Helper function to update the Text widget (Game Log)"""
        self.log_text.config(state=tk.NORMAL) # Enable editing
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END) # Auto-scroll to the bottom
        self.log_text.config(state=tk.DISABLED) # Disable editing

    def start_game(self):
        """Initializes the game state."""
        self.target_number = random.randint(1, 10)
        self.attempts_left = self.max_attempts
        
        # Clear the log and reset status
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        self.status_label.config(text=f"Attempts left: {self.attempts_left}")
        self.log("New game started! Guess the number between 1 and 10!")
        
        self.guess_button.config(state=tk.NORMAL)
        self.guess_entry.config(state=tk.NORMAL)
        self.guess_entry.delete(0, tk.END)

    def check_guess(self):
        """Handles the logic when the 'Guess!' button is pressed."""
        guess_str = self.guess_entry.get()
        self.guess_entry.delete(0, tk.END) # Clear the input box

        try:
            guess = int(guess_str)
            if not (1 <= guess <= 10):
                self.log("⚠️ Please enter a number between 1 and 10!")
                return
            
            self.attempts_left -= 1
            
            if guess == self.target_number:
                self.log(f"🎉 Congrats! You guessed the correct number: {self.target_number}!")
                self.end_game(True)
            else:
                hint = "Lower!" if guess > self.target_number else "Higher!"
                self.log(f"❌ Wrong guess! It's {hint}")
                
                if self.attempts_left <= 0:
                    self.log(f"Game Over! The number was {self.target_number}.")
                    self.end_game(False)
                else:
                    self.status_label.config(text=f"Attempts left: {self.attempts_left}")

        except ValueError:
            self.log("⚠️ Please enter a valid number!")

    def end_game(self, won):
        """Disables input and updates status at the end of a round."""
        self.guess_button.config(state=tk.DISABLED)
        self.guess_entry.config(state=tk.DISABLED)
        self.status_label.config(text="Game Ended. Start a New Game.")


# --- Word Guessing Game GUI Frame (Hangman) ---
class WordGuessingGUI(tk.Frame):
    def __init__(self, parent, controller):
        # Set the frame background color
        tk.Frame.__init__(self, parent, bg="#90EE90")
        self.controller = controller
        
        # Game State Variables
        self.secret_word = ""
        self.guessed_letters = []
        self.attempts = 0
        self.letter_buttons = {} # Dictionary to hold the A-Z buttons

        # --- Widgets Setup ---
        tk.Label(self, text="=== Word Guessing (Hangman) ===", font=('Arial', 14, 'bold'), bg="#90EE90").pack(pady=10)
        
        # Label to display the masked word (e.g., P _ T H O N)
        self.word_display = tk.Label(self, text="", font=('Courier', 24, 'bold'), bg="#90EE90")
        self.word_display.pack(pady=10)
        
        # Label to display attempts left
        self.status_label = tk.Label(self, text="Attempts left: 6", bg="#90EE90")
        self.status_label.pack(pady=5)
        
        # Frame for the Letter Buttons
        self.letter_frame = tk.Frame(self, bg="#90EE90")
        self.letter_frame.pack(pady=20)
        
        self.create_letter_buttons()

        # Control Buttons
        tk.Button(self, text="New Game", command=self.start_game).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(self, text="Back to Menu", command=self.back_to_menu).pack(side=tk.RIGHT, padx=10, pady=10)

        # Start the first game
        self.start_game()

    def create_letter_buttons(self):
        """Creates 26 buttons for A-Z in the letter_frame."""
        for i in range(26):
            letter = chr(ord('a') + i) # 'a', 'b', 'c', ...
            # Create the button, passing the letter to the check_letter method
            button = tk.Button(
                self.letter_frame, 
                text=letter.upper(), 
                width=2, 
                command=lambda l=letter: self.check_letter(l)
            )
            # Layout the buttons in rows of 9
            row = i // 9
            col = i % 9
            button.grid(row=row, column=col, padx=2, pady=2)
            self.letter_buttons[letter] = button

    def start_game(self):
        """Resets the game state and UI for a new round."""
        
        # === MODIFICATION IS HERE ===
        words = load_hangman_words() # Use the new function to load words
        if words == ["error"]:
            self.secret_word = "error"
            self.status_label.config(text="FATAL ERROR: Could not load word list.", fg="red")
            self.disable_all_letters()
            return

        self.secret_word = random.choice(words).lower()
        # ============================
        
        self.guessed_letters = []
        self.attempts = 6
        
        # Reset all letter buttons to be enabled
        for letter, button in self.letter_buttons.items():
            button.config(state=tk.NORMAL)
        
        self.update_display()
        self.status_label.config(text=f"Attempts left: {self.attempts}", fg="black")

    def get_masked_word(self):
        """Generates the string with underscores for unguessed letters."""
        display_word = ""
        for letter in self.secret_word:
            if letter in self.guessed_letters:
                display_word += letter + " "
            else:
                display_word += "_ "
        return display_word.strip()

    def update_display(self):
        """Updates the word display and checks for win/loss conditions."""
        masked_word = self.get_masked_word()
        self.word_display.config(text=masked_word)

        if "_" not in masked_word:
            self.status_label.config(text="🎉 CONGRATULATIONS! You guessed the word!")
            self.disable_all_letters()
        elif self.attempts <= 0:
            self.status_label.config(text=f"❌ GAME OVER! The word was: {self.secret_word}")
            self.disable_all_letters()
        else:
            self.status_label.config(text=f"Attempts left: {self.attempts}")

    def check_letter(self, guess):
        """Handles the logic when a letter button is pressed."""
        
        # Disable the button immediately
        self.letter_buttons[guess].config(state=tk.DISABLED)
        
        if guess not in self.guessed_letters:
            self.guessed_letters.append(guess)

            if guess not in self.secret_word:
                self.attempts -= 1
                
        self.update_display()
        
    def disable_all_letters(self):
        """Disables all letter buttons at the end of the game."""
        for letter, button in self.letter_buttons.items():
            button.config(state=tk.DISABLED)
            
    def back_to_menu(self):
        """Resets the game and returns to the main menu."""
        self.start_game()
        self.controller.show_frame("MainMenu")

# --- Rock-Paper-Scissors GUI Frame ---
class RockPaperScissorsGUI(tk.Frame):
    def __init__(self, parent, controller):
        # Set the frame background color
        tk.Frame.__init__(self, parent, bg="#90EE90")
        self.controller = controller
        
        # Game State Variables
        self.choices = ["rock", "paper", "scissors"]
        self.player_score = 0
        self.computer_score = 0

        # --- Widgets Setup ---
        tk.Label(self, text="=== Rock-Paper-Scissors ===", font=('Arial', 16, 'bold'), bg="#90EE90").pack(pady=10)
        
        # Label for displaying the current score
        self.score_label = tk.Label(self, text="First to 3 wins!\nScore: You 0 - 0 Computer", bg="#90EE90", font=('Arial', 12))
        self.score_label.pack(pady=10)

        # Label for displaying round results and status messages
        self.result_label = tk.Label(self, text="Choose your weapon!", bg="#90EE90", font=('Arial', 12, 'italic'))
        self.result_label.pack(pady=10)

        # Frame for the Choice Buttons
        button_frame = tk.Frame(self, bg="#90EE90")
        button_frame.pack(pady=20)
        
        # Create a button for each choice, mapping the choice to the play_round method
        self.rock_button = tk.Button(button_frame, text="✊ ROCK", width=10, command=lambda: self.play_round("rock"))
        self.rock_button.pack(side=tk.LEFT, padx=10)

        self.paper_button = tk.Button(button_frame, text="✋ PAPER", width=10, command=lambda: self.play_round("paper"))
        self.paper_button.pack(side=tk.LEFT, padx=10)

        self.scissors_button = tk.Button(button_frame, text="✌️ SCISSORS", width=10, command=lambda: self.play_round("scissors"))
        self.scissors_button.pack(side=tk.LEFT, padx=10)
        
        # Control Buttons
        self.new_game_button = tk.Button(self, text="New Game", command=self.reset_game)
        self.new_game_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        tk.Button(self, text="Back to Menu", command=lambda: [self.reset_game(), controller.show_frame("MainMenu")]).pack(side=tk.RIGHT, padx=10, pady=10)

    def update_score_display(self):
        """Updates the Label with the current scores."""
        self.score_label.config(text=f"First to 3 wins!\nScore: You {self.player_score} - {self.computer_score} Computer")

    def toggle_buttons(self, state):
        """Enables or disables the choice buttons."""
        self.rock_button.config(state=state)
        self.paper_button.config(state=state)
        self.scissors_button.config(state=state)

    def play_round(self, player_choice):
        """Contains the core Rock-Paper-Scissors logic."""
        computer_choice = random.choice(self.choices)
        round_result = ""
        
        self.result_label.config(text=f"You chose: {player_choice.upper()} | Computer chose: {computer_choice.upper()}")

        if player_choice == computer_choice:
            round_result = "🤝 It's a tie!"
        elif (
            (player_choice == "rock" and computer_choice == "scissors") or
            (player_choice == "paper" and computer_choice == "rock") or
            (player_choice == "scissors" and computer_choice == "paper")
        ):
            round_result += "🎉 You win this round!"
            self.player_score += 1
        else:
            round_result += "❌ Computer wins this round!"
            self.computer_score += 1

        # Update the result label to show the outcome
        self.result_label.config(text=self.result_label.cget("text") + "\n" + round_result)
        
        self.update_score_display()

        # Check for game end condition (first to 3)
        if self.player_score >= 3:
            self.result_label.config(text="🏆 You won the game! Click 'New Game' to restart.")
            self.toggle_buttons(tk.DISABLED)
        elif self.computer_score >= 3:
            self.result_label.config(text="💻 Computer won the game! Click 'New Game' to restart.")
            self.toggle_buttons(tk.DISABLED)

    def reset_game(self):
        """Resets the scores and allows a new game to begin."""
        self.player_score = 0
        self.computer_score = 0
        self.update_score_display()
        self.result_label.config(text="Choose your weapon!")
        self.toggle_buttons(tk.NORMAL)

# --- Higher or Lower GUI Frame ---
class HigherOrLowerGUI(tk.Frame):
    def __init__(self, parent, controller):
        # Set the frame background color
        tk.Frame.__init__(self, parent, bg="#90EE90")
        self.controller = controller
        
        # Game State Variables
        self.target_number = 0
        self.attempts = 0

        # --- Widgets Setup ---
        tk.Label(self, text="=== Higher or Lower (1-100) ===", font=('Arial', 16, 'bold'), bg="#90EE90").pack(pady=10)
        
        # A Text widget to act as the Game Log
        self.log_text = tk.Text(self, height=8, width=60, state=tk.DISABLED)
        self.log_text.pack(pady=10)

        # Label to show the status (Attempt count)
        self.status_label = tk.Label(self, text="Attempts: 0", bg="#90EE90", font=('Arial', 12))
        self.status_label.pack(pady=5)

        # Frame for Input and Button
        input_frame = tk.Frame(self, bg="#90EE90")
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="Your Guess (1-100):", bg="#90EE90").pack(side=tk.LEFT, padx=5)
        
        # Entry widget for the player's guess
        self.guess_entry = tk.Entry(input_frame, width=10)
        self.guess_entry.pack(side=tk.LEFT, padx=5)
        
        # Button to submit the guess
        self.guess_button = tk.Button(input_frame, text="Guess!", command=self.check_guess)
        self.guess_button.pack(side=tk.LEFT, padx=5)

        # Control Buttons
        tk.Button(self, text="New Game", command=self.start_game).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(self, text="Back to Menu", command=lambda: [self.start_game(), controller.show_frame("MainMenu")]).pack(side=tk.RIGHT, padx=10, pady=10)

        # Start the first game automatically
        self.start_game()
        
    def log(self, message):
        """Helper function to update the Text widget (Game Log)"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def start_game(self):
        """Initializes the game state."""
        self.target_number = random.randint(1, 100)
        self.attempts = 0
        
        # Clear the log and reset status
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        self.status_label.config(text="Attempts: 0")
        self.log("New game started! Guess the number between 1 and 100!")
        
        self.guess_button.config(state=tk.NORMAL)
        self.guess_entry.config(state=tk.NORMAL)
        self.guess_entry.delete(0, tk.END)

    def check_guess(self):
        """Handles the logic when the 'Guess!' button is pressed."""
        guess_str = self.guess_entry.get()
        self.guess_entry.delete(0, tk.END) # Clear the input box

        try:
            guess = int(guess_str)
            if not (1 <= guess <= 100):
                self.log("⚠️ Please enter a number between 1 and 100!")
                return
            
            self.attempts += 1
            self.status_label.config(text=f"Attempts: {self.attempts}")
            
            if guess == self.target_number:
                self.log(f"🎉 Correct! You guessed it ({self.target_number}) in {self.attempts} attempts!")
                self.end_game()
            elif guess < self.target_number:
                self.log(f"Your guess: {guess}. ⬆️ Higher!")
            else:
                self.log(f"Your guess: {guess}. ⬇️ Lower!")

        except ValueError:
            self.log("⚠️ Please enter a valid number!")

    def end_game(self):
        """Disables input and updates status at the end of a round."""
        self.guess_button.config(state=tk.DISABLED)
        self.guess_entry.config(state=tk.DISABLED)

# --- Dice Rolling Game GUI Frame (Betting) ---
class DiceRollingGUI(tk.Frame):
    def __init__(self, parent, controller):
        # Set the frame background color
        tk.Frame.__init__(self, parent, bg="#90EE90")
        self.controller = controller
        
        # Game State Variables
        self.money = 100
        self.bet_choice = tk.StringVar(value="h") # Tkinter variable for Radiobuttons (default to High)

        # --- Widgets Setup ---
        tk.Label(self, text="=== Dice Rolling Game ===", font=('Arial', 16, 'bold'), bg="#90EE90").pack(pady=10)
        
        # Label for displaying current money
        self.money_label = tk.Label(self, text=f"💰 Current Money: ${self.money}", bg="#90EE90", font=('Arial', 14, 'bold'))
        self.money_label.pack(pady=10)

        # A Text widget to act as the Game Log (dice rolls and results)
        self.log_text = tk.Text(self, height=8, width=60, state=tk.DISABLED)
        self.log_text.pack(pady=10)

        # Frame for Betting Input
        input_frame = tk.Frame(self, bg="#90EE90")
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="Bet Amount:", bg="#90EE90").pack(side=tk.LEFT, padx=5)
        
        # Entry widget for the bet amount
        self.bet_entry = tk.Entry(input_frame, width=10)
        self.bet_entry.pack(side=tk.LEFT, padx=10)

        # Radiobuttons for Choice (High or Low)
        tk.Radiobutton(input_frame, text="High (4-6)", variable=self.bet_choice, value="h", bg="#90EE90").pack(side=tk.LEFT)
        tk.Radiobutton(input_frame, text="Low (1-3)", variable=self.bet_choice, value="l", bg="#90EE90").pack(side=tk.LEFT)
        
        # Button to place the bet
        self.place_bet_button = tk.Button(self, text="Place Bet!", command=self.place_bet)
        self.place_bet_button.pack(pady=10)

        # Control Buttons
        tk.Button(self, text="New Game (Reset Money)", command=self.reset_game).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(self, text="Back to Menu", command=lambda: [self.reset_game(), controller.show_frame("MainMenu")]).pack(side=tk.RIGHT, padx=10, pady=10)

        self.log("Welcome! Bet on High (4-6) or Low (1-3)!")

    def log(self, message):
        """Helper function to update the Text widget (Game Log)"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def update_money_display(self):
        """Updates the money label."""
        self.money_label.config(text=f"💰 Current Money: ${self.money}")

    def place_bet(self):
        """Handles the dice roll and betting logic."""
        bet_str = self.bet_entry.get()
        self.bet_entry.delete(0, tk.END) # Clear the input box

        try:
            bet = int(bet_str)
            
            # Validation checks
            if bet <= 0:
                self.log("⚠️ Bet must be a positive number!")
                return
            if bet > self.money:
                self.log(f"⚠️ You only have ${self.money}! Bet less than that.")
                return
            
            choice = self.bet_choice.get() # 'h' or 'l'
            dice = random.randint(1, 6)
            
            self.log(f"\n--- Round ---")
            self.log(f"Bet: ${bet} on {'HIGH' if choice == 'h' else 'LOW'}")
            self.log(f"🎲 Dice rolled: {dice}")

            # Check win condition
            is_win = (choice == "h" and dice >= 4) or (choice == "l" and dice <= 3)

            if is_win:
                self.money += bet
                self.log(f"✅ YOU WIN! You gained ${bet}.")
            else:
                self.money -= bet
                self.log(f"❌ You lose ${bet}.")
            
            self.update_money_display()

            # Check for Game Over
            if self.money <= 0:
                self.log("\n💸 GAME OVER! You're out of money. Start a New Game.")
                self.place_bet_button.config(state=tk.DISABLED)
                self.bet_entry.config(state=tk.DISABLED)

        except ValueError:
            self.log("⚠️ Please enter a valid numerical bet!")

    def reset_game(self):
        """Resets money to 100 for a new game."""
        self.money = 100
        self.update_money_display()
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        self.log("New game started. Bet wisely!")
        self.place_bet_button.config(state=tk.NORMAL)
        self.bet_entry.config(state=tk.NORMAL)

# --- Quiz Selection GUI Frame ---
class QuizSelectionGUI(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#90EE90")
        self.controller = controller
        
        # --- Widgets Setup ---
        tk.Label(self, text="=== Select Quiz Difficulty ===", font=('Arial', 16, 'bold'), bg="#90EE90").pack(pady=20)
        
        tk.Label(self, text="How many questions would you like to answer?", font=('Arial', 12), bg="#90EE90").pack(pady=10)
        
        # Frame for the difficulty buttons
        button_frame = tk.Frame(self, bg="#90EE90")
        button_frame.pack(pady=30)
        
        tk.Button(button_frame, text="10 Questions", width=15, font=('Arial', 12), command=lambda: self.start_quiz(10)).pack(pady=10)
        tk.Button(button_frame, text="20 Questions", width=15, font=('Arial', 12), command=lambda: self.start_quiz(20)).pack(pady=10)
        tk.Button(button_frame, text="30 Questions", width=15, font=('Arial', 12), command=lambda: self.start_quiz(30)).pack(pady=10)
        
        # Control Buttons
        tk.Button(self, text="Back to Menu", command=lambda: controller.show_frame("MainMenu")).pack(pady=20)
    
    def start_quiz(self, num_questions):
        """Transitions to the quiz game with the selected number of questions."""
        quiz_frame = self.controller.frames["QuizGameGUI"]
        quiz_frame.set_question_count(num_questions)
        quiz_frame.start_game()
        self.controller.show_frame("QuizGameGUI")

# --- Quiz Game GUI Frame ---
class QuizGameGUI(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#90EE90")
        self.controller = controller
        
        # Game State Variables
        self.questions = []
        self.current_question_index = 0
        self.score = 0
        self.num_questions = 10  # Default to 10 questions

        # --- Widgets Setup ---
        tk.Label(self, text="=== Quiz Game ===", font=('Arial', 16, 'bold'), bg="#90EE90").pack(pady=10)
        
        # Label to display the current score
        self.score_label = tk.Label(self, text="Score: 0/0", bg="#90EE90", font=('Arial', 12))
        self.score_label.pack(pady=5)

        # Label for the question text
        self.question_label = tk.Label(self, text="", wraplength=500, justify=tk.LEFT, bg="#90EE90", font=('Arial', 14))
        self.question_label.pack(pady=20, padx=10)

        # Frame for Answer Input
        input_frame = tk.Frame(self, bg="#90EE90")
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="Your Answer:", bg="#90EE90").pack(side=tk.LEFT, padx=5)
        
        self.answer_entry = tk.Entry(input_frame, width=30)
        self.answer_entry.pack(side=tk.LEFT, padx=5)
        
        self.submit_button = tk.Button(input_frame, text="Submit", command=self.submit_answer)
        self.submit_button.pack(side=tk.LEFT, padx=5)
        
        # Label for feedback (Correct/Wrong)
        self.feedback_label = tk.Label(self, text="", bg="#90EE90", font=('Arial', 12, 'italic'))
        self.feedback_label.pack(pady=10)

        # Control Buttons
        tk.Button(self, text="New Game", command=lambda: controller.show_frame("QuizSelectionGUI")).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(self, text="Back to Menu", command=lambda: controller.show_frame("MainMenu")).pack(side=tk.RIGHT, padx=10, pady=10)
    
    def set_question_count(self, num_questions):
        """Sets the number of questions to use for the upcoming game."""
        self.num_questions = num_questions
    
    def start_game(self):
        """Loads questions and resets the game state."""
        all_questions = load_quiz_questions() # Load all questions from the JSON file
        
        # Randomly select only the specified number of questions
        if len(all_questions) >= self.num_questions:
            self.questions = random.sample(all_questions, self.num_questions)
        else:
            self.questions = all_questions  # Use all if there aren't enough
        
        self.current_question_index = 0
        self.score = 0
        
        self.submit_button.config(state=tk.NORMAL)
        self.answer_entry.config(state=tk.NORMAL)
        self.feedback_label.config(text="")
        
        if self.questions:
            self.show_next_question()
        else:
            self.question_label.config(text="No questions loaded. Check quiz_data.json.")
            self.submit_button.config(state=tk.DISABLED)

    def show_next_question(self):
        """Displays the next question in the list."""
        if self.current_question_index < len(self.questions):
            q_data = self.questions[self.current_question_index]
            self.question_label.config(text=f"Q{self.current_question_index + 1}: {q_data['question']}")
            self.score_label.config(text=f"Score: {self.score}/{len(self.questions)}")
            self.answer_entry.delete(0, tk.END) # Clear previous answer
            self.feedback_label.config(text="")
        else:
            self.end_game()

    def submit_answer(self):
        """Checks the player's answer against the correct answer."""
        if self.current_question_index >= len(self.questions):
            return # Should not happen, but a safety check
            
        q_data = self.questions[self.current_question_index]
        user_answer = normalize_answer(self.answer_entry.get())
        correct_answer = normalize_answer(q_data['answer'])
        
        if user_answer == correct_answer:
            self.score += 1
            self.feedback_label.config(text="✅ Correct!", fg="green")
        else:
            self.feedback_label.config(text=f"❌ Wrong! Answer: {q_data['answer']}", fg="red")
            # Add info if available
            if 'info' in q_data:
                self.feedback_label.config(text=f"❌ Wrong! Answer: {q_data['answer']}\n{q_data['info']}")
        
        # Move to the next question after a brief delay so the user can see the feedback
        self.current_question_index += 1
        self.after(1000, self.show_next_question)

    def end_game(self):
        """Displays final score."""
        final_score_text = f"📊 Quiz Finished! Your final score is: {self.score}/{len(self.questions)}"
        self.question_label.config(text=final_score_text)
        self.score_label.config(text=f"Final Score: {self.score}/{len(self.questions)}")
        self.submit_button.config(state=tk.DISABLED)
        self.answer_entry.config(state=tk.DISABLED)

# --- Tic-Tac-Toe Selection GUI Frame ---
class TicTacToeSelectionGUI(tk.Frame):
    """Screen to select 1 player or 2 player mode."""
    def __init__(self, parent, controller):
        bg="#90EE90"
        tk.Frame.__init__(self, parent, bg=bg) 
        self.controller = controller

        tk.Label(self, text="=== Tic-Tac-Toe Mode Selection ===", font=('Arial', 16, 'bold'), bg=bg).pack(pady=20)
        
        tk.Label(self, text="Choose your game mode:", font=('Arial', 12), bg=bg).pack(pady=10)
        
        tk.Button(self, text="1 Player (vs Computer)", width=30, font=('Arial', 12), command=self.start_one_player).pack(pady=10)
        tk.Button(self, text="2 Players (Local)", width=30, font=('Arial', 12), command=self.start_two_player).pack(pady=10)
        
        tk.Button(self, text="Back to Menu", width=30, command=lambda: controller.show_frame("MainMenu")).pack(pady=20)
    
    def start_one_player(self):
        """Starts 1 player game."""
        tictactoe_frame = self.controller.frames["TicTacToeGUI"]
        tictactoe_frame.set_game_mode(one_player=True)
        self.controller.show_frame("TicTacToeGUI")
    
    def start_two_player(self):
        """Starts 2 player game."""
        tictactoe_frame = self.controller.frames["TicTacToeGUI"]
        tictactoe_frame.set_game_mode(one_player=False)
        self.controller.show_frame("TicTacToeGUI")

# --- Tic-Tac-Toe GUI Frame ---
class TicTacToeGUI(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#90EE90")
        self.controller = controller
        
        # Game State Variables
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = "X"
        self.buttons = {}  # Dictionary to hold the 9 button objects
        self.game_active = True
        self.one_player = False  # Whether it's 1 player (vs computer) or 2 player
        self.ai_thinking = False  # Flag to prevent clicks while computer is thinking

        # --- Widgets Setup ---
        self.title_label = tk.Label(self, text="=== Tic-Tac-Toe (2-Player) ===", font=('Arial', 16, 'bold'), bg="#90EE90")
        self.title_label.pack(pady=10)
        
        # Label to display current status
        self.status_label = tk.Label(self, text="Player X's turn", bg="#90EE90", font=('Arial', 14))
        self.status_label.pack(pady=10)

        # Frame for the 3x3 Game Board Grid
        grid_frame = tk.Frame(self, bg="#90EE90")
        grid_frame.pack(pady=20)
        
        # Create 9 buttons in a 3x3 grid
        for row in range(3):
            for col in range(3):
                # We use a lambda to pass the row and col arguments to the button_click function
                button = tk.Button(
                    grid_frame, 
                    text=" ", 
                    font=('Arial', 24, 'bold'), 
                    width=4, 
                    height=2,
                    command=lambda r=row, c=col: self.button_click(r, c)
                )
                button.grid(row=row, column=col, padx=2, pady=2)
                self.buttons[(row, col)] = button # Store button reference by (row, col)

        # Control Buttons
        tk.Button(self, text="New Game", command=self.reset_game).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(self, text="Back to Selection", command=lambda: [self.reset_game(), controller.show_frame("TicTacToeSelectionGUI")]).pack(side=tk.RIGHT, padx=10, pady=10)

    def set_game_mode(self, one_player):
        """Sets whether it's 1 player or 2 player mode."""
        self.one_player = one_player
        mode_text = "1 Player (vs Computer)" if one_player else "2 Player (Local)"
        self.title_label.config(text=f"=== Tic-Tac-Toe: {mode_text} ===")
        self.reset_game()
    
    def get_available_moves(self):
        """Returns a list of available board positions."""
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves
    
    def evaluate_board(self):
        """Simple board evaluation: +10 for X win, -10 for O win, 0 otherwise."""
        winner = self.check_winner()
        if winner == 'X':
            return 10
        elif winner == 'O':
            return -10
        else:
            return 0
    
    def minimax(self, depth, is_maximizing):
        """Minimax algorithm for AI (simple depth-limited search)."""
        score = self.evaluate_board()
        
        # Terminal states
        if score == 10:
            return score - depth
        if score == -10:
            return score + depth
        
        available_moves = self.get_available_moves()
        if not available_moves:
            return 0
        
        # Limit depth to avoid long thinking time
        if depth >= 6:
            return 0
        
        if is_maximizing:
            max_score = -float('inf')
            for row, col in available_moves:
                self.board[row][col] = 'X'
                score = self.minimax(depth + 1, False)
                self.board[row][col] = ' '
                max_score = max(score, max_score)
            return max_score
        else:
            min_score = float('inf')
            for row, col in available_moves:
                self.board[row][col] = 'O'
                score = self.minimax(depth + 1, True)
                self.board[row][col] = ' '
                min_score = min(score, min_score)
            return min_score
    
    def get_best_move(self):
        """Gets the best move for the AI using minimax algorithm."""
        best_score = -float('inf')
        best_move = None
        
        for row, col in self.get_available_moves():
            self.board[row][col] = 'O'
            score = self.minimax(0, False)
            self.board[row][col] = ' '
            
            if score > best_score:
                best_score = score
                best_move = (row, col)
        
        return best_move
    
    def computer_move(self):
        """Makes the computer's move."""
        self.ai_thinking = True
        self.status_label.config(text="Computer is thinking...")
        self.update()
        
        move = self.get_best_move()
        if move:
            row, col = move
            self.board[row][col] = 'O'
            self.buttons[(row, col)].config(text='O', state=tk.DISABLED)
            
            winner = self.check_winner()
            if winner:
                self.status_label.config(text=f"🎉 Player {winner} wins!")
                self.game_active = False
                self.disable_all_buttons()
            elif not self.get_available_moves():
                self.status_label.config(text="🤝 It's a tie!")
                self.game_active = False
            else:
                self.current_player = "X"
                self.status_label.config(text="Player X's turn")
        
        self.ai_thinking = False

    def check_winner(self):
        """Checks the 3x3 board for a winner."""
        # Check rows, columns, and diagonals using the board array
        
        # Check rows
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != ' ':
                return self.board[i][0]
        
        # Check columns
        for j in range(3):
            if self.board[0][j] == self.board[1][j] == self.board[2][j] != ' ':
                return self.board[0][j]
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
        
        return None

    def button_click(self, row, col):
        """Handles the logic when a grid button is pressed."""
        if not self.game_active or self.board[row][col] != ' ' or self.ai_thinking:
            return

        # Only allow clicks on X's turn in 1 player mode
        if self.one_player and self.current_player != 'X':
            return

        # 1. Update the board state
        self.board[row][col] = self.current_player
        
        # 2. Update the button text and disable it
        button = self.buttons[(row, col)]
        button.config(text=self.current_player, state=tk.DISABLED)
        
        winner = self.check_winner()

        if winner:
            self.status_label.config(text=f"🎉 Player {winner} wins!")
            self.game_active = False
            self.disable_all_buttons()
        
        # Check for a tie
        elif all(self.board[i][j] != ' ' for i in range(3) for j in range(3)):
            self.status_label.config(text="🤝 It's a tie!")
            self.game_active = False

        else:
            # 3. Switch player and update status label
            self.current_player = "O" if self.current_player == "X" else "X"
            
            # If 1 player mode and it's now computer's turn, make the computer move
            if self.one_player and self.current_player == "O":
                self.after(500, self.computer_move)  # Delay 500ms for better UX
            else:
                self.status_label.config(text=f"Player {self.current_player}'s turn")

    def disable_all_buttons(self):
        """Disables all buttons at the end of the game."""
        for button in self.buttons.values():
            if button['text'] == ' ': # Don't re-disable buttons already played
                button.config(state=tk.DISABLED)
                
    def reset_game(self):
        """Resets the board and game state."""
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = "X"
        self.game_active = True
        self.ai_thinking = False
        self.status_label.config(text="Player X's turn")
        
        # Reset all button appearances and state
        for button in self.buttons.values():
            button.config(text=" ", state=tk.NORMAL)

# --- Execution ---
if __name__ == "__main__":
    app = GameApp()
    app.mainloop()