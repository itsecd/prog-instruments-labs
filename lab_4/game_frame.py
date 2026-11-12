"""Base class for game screens that share a read-only log widget."""
import tkinter as tk


class GameFrame(tk.Frame):
    """Frame with a read-only ``log_text`` widget and helpers to write to it.

    Subclasses must create ``self.log_text`` (a ``tk.Text`` widget).
    """

    def log(self, message):
        """Append a message to the game log and scroll to the bottom."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def clear_log(self):
        """Remove all messages from the game log."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state=tk.DISABLED)