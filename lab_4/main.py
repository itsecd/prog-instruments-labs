from tkinter import *
from tkinter import ttk, filedialog


# This will be received everytime something new have been typed in, "a" is only a placeholder
def update_i(_):
    global i
    i = textarea.get("1.0", "end-1c")
    initialize_data()


# Initialize the data for the search in the query
def initialize_data():
    global i
    v = vect_int(word_vect(i))
    tp = ind.get(v, 'Word Not in Dictionary.')
    word_listbox.delete(0, END)
    if tp == 'Word Not in Dictionary.':
        word_listbox.insert(END, tp)
    else:
        for i in range(0, len(tp)):
            word_listbox.insert(END, tp[i])


# Open the dictionary file and read it
def open_dict():
    global dict_open, y, raw, ind
    file = filedialog.askopenfilename(initialdir="./",
                                      title="Open a text file",
                                      filetypes=(("text", "*.txt"), ("all", "*.*")))
    try:
        dict_open = open(file, "r")
        raw = dict_open.read()
        dict_open.close()
        if y is False:
            dict_list = raw.split(DEFAULT_SEPERATOR)
            ind = int_dict(dict_list)
        else:
            new_seperator()
    except FileNotFoundError:
        print("Not Found, try getting the dictionary")
        return
    except NameError:
        print("NameError")
        return
    finally:
        return dict_list


# Change the seperator incase the dictionary has a different seperator
def new_seperator():
    global y, ind
    y = True
    dict_list = raw.split(new_seperator_entry.get())
    seperator_label.config(text=f'Seperator: "{new_seperator_entry.get()}"')
    ind = int_dict(dict_list)
    initialize_data()


# Set back to the original seperator
def default_seperator():
    global ind
    dict_list = raw.split(DEFAULT_SEPERATOR)
    ind = int_dict(dict_list)
    seperator_label.config(text=f'Seperator: "{DEFAULT_SEPERATOR}"')


# Count the amount of characters and their duplicates appearing
def word_vect(word):
    alphabet_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                     'u', 'v', 'w', 'x', 'y', 'z']
    dupe_count = [0 for _ in range(26)]
    wrd = word.lower()
    word_list = list(wrd)
    for temp in range(0, len(word_list)):
        if word_list[temp] in alphabet_list:
            index = alphabet_list.index(word_list[temp])
            dupe_count[index] += 1
    return dupe_count


# convert it into integer
def vect_int(vect):
    counter = num = 0
    for i in range(0, len(vect)):
        clt = (vect[i] * (2 ** counter))
        num += clt
        counter += 4
    return num


# make the dictionary
def int_dict(dic):
    dictn = {}
    for i in range(0, len(dic)):
        v = word_vect(dic[i])
        Int = vect_int(v)
        if Int in dictn:
            tat = dictn.get(Int)
            tat.append(dic[i])
            dictn[Int] = tat
        elif Int not in dictn:
            dictn[Int] = [dic[i]]
    return dictn


# Check the dictionary name to make sure it is correct
def check_constant(_):
    file_label.config(text="File: " + f"{dict_open}".split("'")[1].split("/")[-1])


# let the prograsm stay on top of other apps
def on_top():
    global top
    if top is False:
        root.attributes('-topmost', True)
        root.update()
        on_top_button.config(relief=SUNKEN)
        top = True
    else:
        root.attributes('-topmost', False)
        root.update()
        on_top_button.config(relief=RAISED)
        top = False


if __name__ == '__main__':
    # Initialization
    x = y = top = False
    DEFAULT_SEPERATOR = '\n'
    dictn = open_dict()
    ind = int_dict(dictn)

    # main Window
    root = Tk()
    root.title('Deciphr')
    root.resizable(width=False, height=False)

    # Set up 2 tabs for the window
    notebook = ttk.Notebook(root)

    tab1 = Frame(notebook)
    tab2 = Frame(notebook)

    notebook.add(tab1, text="Finder")
    notebook.add(tab2, text="File")

    # Tab 1 Widgets
    textarea = Text(tab1,
                    font=("Arial", 15),
                    height=1,
                    width=20)

    word_listbox = Listbox(tab1,
                           font=("Arial", 15))

    # Tab 2 widgets
    openbutton = Button(tab2,
                        text="Open new Dictionary",
                        font=("Arial", 8),
                        command=open_dict)

    file_label = Label(tab2,
                       text="",
                       font=("Arial", 8))

    seperator_label = Label(tab2,
                            text=f'Seperator: "{DEFAULT_SEPERATOR}"')

    new_seperator_entry = Entry(tab2,
                                font=("Arial", 8))

    seperator_frame = Frame(tab2)

    new_seperator_button = Button(seperator_frame,
                                  text="Set as Seperator",
                                  font=("Arial", 8),
                                  command=new_seperator)

    default_seperator_button = Button(seperator_frame,
                                      text="Default Seperator",
                                      font=("Arial", 8),
                                      command=default_seperator)

    credit_label = Label(tab2,
                         text="""Creator: Baguette\nCredit:\n- CS50 Staff\n""",
                         font=("Arial", 8, "italic"))

    on_top_button = Button(tab2,
                           text="Stay on top",
                           font=("Arial", 8),
                           command=on_top)

    # detect any new text
    textarea.bind("<KeyRelease>", update_i)

    # detect any mouse clicks
    root.bind("<Button-1>", check_constant)

    # pack the widgets
    # tab 1
    textarea.pack()
    word_listbox.pack()

    # tab2
    openbutton.pack()
    file_label.pack()
    seperator_label.pack()
    new_seperator_entry.pack()
    seperator_frame.pack()
    new_seperator_button.pack(side=LEFT)
    default_seperator_button.pack(side=RIGHT)

    # Add a space for aesthetic reasons
    Label(tab2).pack()
    credit_label.pack()
    on_top_button.pack()
    notebook.pack(expand=True, fill="both")
    file_label.config(text="File: " + f"{dict_open}".split("'")[1].split("/")[-1])
    root.mainloop()
