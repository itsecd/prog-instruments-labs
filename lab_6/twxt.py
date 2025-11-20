# Importing required modules
from tkinter import *
from tkinter import filedialog, colorchooser, font, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText
import os


class MmabiaaTextpad:
    def __init__(self, root=None):
        if root is None:
            self.root = Tk()
            self.root.title("Mmabiaa Textpad")
            self.root.geometry("800x600")
        else:
            self.root = root

        # Initialize the global filename and font variables
        self.filename = None
        self.current_font_family = "Times New Roman"
        self.current_font_size = 18
        self.text_area = None

        self.create_menu()
        self.create_text_area()

        # Only run mainloop if we created the root window
        if root is None:
            self.root.mainloop()

    # Function to create a new file
    def new_file(self):
        # new_file : clears the text area and resets the filename
        self.text_area.delete(1.0, END)
        self.filename = None
        self.root.title("Mmabia Text Pad- Untitled File")  # Sets the title as Mmabia Textpad

    # Function to open an existing file
    def open_file(self):
        # open_file : opens a file dialog,reads the selected file and displays it contents in the text area
        filename = filedialog.askopenfilename(defaultextension=".txt",
                                              filetypes=[("All Files", "."), ("Text Documents", "*.txt")])
        if filename:
            self.filename = filename
            with open(filename, 'r') as file:
                self.text_area.delete(1.0, END)
                self.text_area.insert(1.0, file.read())
            self.root.title(f"Mmabia Textpad - {os.path.basename(filename)}")

    # Function to save the current file
    def save_file(self):
        # save_file : saves the current file
        if self.filename:
            try:
                with open(self.filename, 'w') as file:
                    file.write(self.text_area.get(1.0, END))
                self.root.title(f"Mmabia Textpad - {os.path.basename(self.filename)}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            self.save_as_file()

    # Function to save the current file with a new name
    def save_as_file(self):
        # save_as_file : opens a save dialog and saves the file with a name
        filename = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=[("All Files", "."), ("Text Documents", "*.txt")])
        if filename:
            self.filename = filename
            with open(filename, 'w') as file:
                file.write(self.text_area.get(1.0, END))
            self.root.title(f"Mmabia Textpad - {os.path.basename(filename)}")

    # Function to choose and set the font
    def choose_font(self):
        # choose_font : opens a font dialog and sets the font
        font_family = simpledialog.askstring("Font", "Enter font family:")
        font_size = simpledialog.askinteger("Font", "Enter font size:")
        if font_family and font_size:
            self.current_font_family = font_family
            self.current_font_size = font_size
            self.text_area.configure(font=(self.current_font_family, self.current_font_size))

    # Function to increase the font size
    def increase_font_size(self):
        # This function helps to increase the font size of a font by 5
        self.current_font_size += 5
        self.text_area.configure(font=(self.current_font_family, self.current_font_size))

    # Function to decrease the font size
    def decrease_font_size(self):
        # This function helps to decrease the font size of a font by 5
        if self.current_font_size > 5:
            self.current_font_size -= 5
            self.text_area.configure(font=(self.current_font_family, self.current_font_size))

    # Function to choose and set the text color
    def choose_color(self):
        # A Function that allows users to choose font color
        color = colorchooser.askcolor()[1]
        if color:
            self.text_area.configure(fg=color)

    # Function to change the theme
    def change_theme(self, theme):
        if theme == "light":
            self.text_area.configure(bg="white", fg="black")
        elif theme == "dark":
            self.text_area.configure(bg="black", fg="white")
        elif theme == "gray":
            self.text_area.configure(bg="grey", fg="white")
        elif theme == "green":
            self.text_area.configure(bg="green", fg="black")
        elif theme == "blue":
            self.text_area.configure(bg="blue", fg="white")
        elif theme == "purple":
            self.text_area.configure(bg="purple", fg="white")
        elif theme == "orange":
            self.text_area.configure(bg="orange", fg="black")
        elif theme == "yellow":
            self.text_area.configure(bg="yellow", fg="black")
        elif theme == "pink":
            self.text_area.configure(bg="pink", fg="black")
        elif theme == "brown":
            self.text_area.configure(bg="brown", fg="white")
        elif theme == "cyan":
            self.text_area.configure(bg="cyan", fg="black")
        elif theme == "magenta":
            self.text_area.configure(bg="magenta", fg="white")
        elif theme == "custom":
            self.text_area.configure(bg="aqua", fg="white")

    # Function to insert an image
    def insert_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", ".png;.jpg;.jpeg;.gif")])
        if filepath:
            try:
                image = PhotoImage(file=filepath)
                self.text_area.image_create(END, image=image)
                self.text_area.image = image  # Keep a reference to avoid garbage collection
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # Function to insert a video (placeholder implementation)
    def insert_video(self):
        filepath = filedialog.askopenfilename(filetypes=[("Video Files", ".mp4;.avi;*.mov")])
        if filepath:
            messagebox.showinfo("Info", "Video inserted. (This is a placeholder implementation.)")

    # Function to apply bold formatting
    def apply_bold(self):
        try:
            current_tags = self.text_area.tag_names("sel.first")
            if "bold" in current_tags:
                self.text_area.tag_remove("bold", "sel.first", "sel.last")
            else:
                self.text_area.tag_add("bold", "sel.first", "sel.last")
                bold_font = font.Font(self.text_area, self.text_area.cget("font"))
                bold_font.configure(weight="bold")
                self.text_area.tag_configure("bold", font=bold_font)
        except TclError:
            pass

    # Function to apply strikethrough formatting
    def apply_strikethrough(self):
        try:
            current_tags = self.text_area.tag_names("sel.first")
            if "strikethrough" in current_tags:
                self.text_area.tag_remove("strikethrough", "sel.first", "sel.last")
            else:
                self.text_area.tag_add("strikethrough", "sel.first", "sel.last")
                strikethrough_font = font.Font(self.text_area, self.text_area.cget("font"))
                strikethrough_font.configure(slant="italic")
                self.text_area.tag_configure("strikethrough", font=strikethrough_font)
        except TclError:
            pass

    # Function to apply italic formatting
    def apply_italic(self):
        try:
            current_tags = self.text_area.tag_names("sel.first")
            if "italic" in current_tags:
                self.text_area.tag_remove("italic", "sel.first", "sel.last")
            else:
                self.text_area.tag_add("italic", "sel.first", "sel.last")
                italic_font = font.Font(self.text_area, self.text_area.cget("font"))
                italic_font.configure(slant="italic")
                self.text_area.tag_configure("italic", font=italic_font)
        except TclError:
            pass

    # Function to apply underline formatting
    def apply_underline(self):
        try:
            current_tags = self.text_area.tag_names("sel.first")
            if "underline" in current_tags:
                self.text_area.tag_remove("underline", "sel.first", "sel.last")
            else:
                self.text_area.tag_add("underline", "sel.first", "sel.last")
                underline_font = font.Font(self.text_area, self.text_area.cget("font"))
                underline_font.configure(underline=True)
                self.text_area.tag_configure("underline", font=underline_font)
        except TclError:
            pass

    # Function for about
    def about(self):
        messagebox.showinfo("About",
                            "Mmabia Text Editor\nVersion 1.0\n\n Created by Boateng Agyenim Prince\n\n A simple text editor built using python and tkinter")

    # Function to create the menu
    def create_menu(self):
        menu = Menu(self.root)
        self.root.config(menu=menu)

        # File menu
        file_menu = Menu(menu, tearoff=False)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_command(label="Save As", command=self.save_as_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Edit menu
        edit_menu = Menu(menu, tearoff=False)
        menu.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=lambda: self.text_area.event_generate("<<Undo>>"))
        edit_menu.add_command(label="Redo", command=lambda: self.text_area.event_generate("<<Redo>>"))
        edit_menu.add_separator()
        edit_menu.add_command(label="Cut", command=lambda: self.text_area.event_generate("<<Cut>>"))
        edit_menu.add_command(label="Copy", command=lambda: self.text_area.event_generate("<<Copy>>"))
        edit_menu.add_command(label="Paste", command=lambda: self.text_area.event_generate("<<Paste>>"))
        edit_menu.add_command(label="Select All", command=lambda: self.text_area.event_generate("<<SelectAll>>"))
        edit_menu.add_separator()

        # Format menu
        format_menu = Menu(menu, tearoff=False)
        menu.add_cascade(label="Format", menu=format_menu)
        format_menu.add_command(label="Font", command=self.choose_font)
        format_menu.add_command(label="Increase Font Size", command=self.increase_font_size)
        format_menu.add_command(label="Decrease Font Size", command=self.decrease_font_size)
        format_menu.add_command(label="Color", command=self.choose_color)
        format_menu.add_separator()
        format_menu.add_command(label="Bold", command=self.apply_bold)
        format_menu.add_command(label="Italic", command=self.apply_italic)
        format_menu.add_command(label="Underline", command=self.apply_underline)
        format_menu.add_command(label="Strikethrough", command=self.apply_strikethrough)

        # Theme menu
        theme_menu = Menu(menu, tearoff=False)
        menu.add_cascade(label="Theme", menu=theme_menu)
        theme_menu.add_command(label="Light", command=lambda: self.change_theme("light"))
        theme_menu.add_command(label="Dark", command=lambda: self.change_theme("dark"))
        theme_menu.add_command(label="Gray", command=lambda: self.change_theme("gray"))
        theme_menu.add_command(label="Green", command=lambda: self.change_theme("green"))
        theme_menu.add_command(label="Blue", command=lambda: self.change_theme("blue"))
        theme_menu.add_command(label="Purple", command=lambda: self.change_theme("purple"))
        theme_menu.add_command(label="Orange", command=lambda: self.change_theme("orange"))
        theme_menu.add_command(label="Yellow", command=lambda: self.change_theme("yellow"))
        theme_menu.add_command(label="Pink", command=lambda: self.change_theme("pink"))
        theme_menu.add_command(label="Brown", command=lambda: self.change_theme("brown"))
        theme_menu.add_command(label="Cyan", command=lambda: self.change_theme("cyan"))
        theme_menu.add_command(label="Magenta", command=lambda: self.change_theme("magenta"))
        theme_menu.add_command(label="Custom", command=lambda: self.change_theme("custom"))

        # Help menu
        help_menu = Menu(menu, tearoff=False)
        menu.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.about)

        # Insert menu
        insert_menu = Menu(menu, tearoff=False)
        menu.add_cascade(label="Insert", menu=insert_menu)
        insert_menu.add_command(label="Image", command=self.insert_image)
        insert_menu.add_command(label="Video", command=self.insert_video)

    # Function to create the text area
    def create_text_area(self):
        self.text_area = ScrolledText(self.root, wrap='word', undo=True,
                                      font=(self.current_font_family, self.current_font_size))
        self.text_area.pack(fill='both', expand=1)
        self.text_area.focus_set()


if __name__ == "__main__":
    app = MmabiaaTextpad()