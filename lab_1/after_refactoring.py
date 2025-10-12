import datetime
import math
import random
import statistics


class LibrarySystem:
    def __init__(self, name, city):
        self.name = name
        self.city = city
        self.catalog = []
        self.members = []
        self.loans = []
        self.history = []
        self.next_book_id = 1
        self.next_member_id = 1

    def add_book(self, title, author, year, copies=1):
        b = Book(self.next_book_id, title, author, year, copies)
        self.next_book_id += 1
        self.catalog.append(b)
        self.history.append(("add", b))
        return b

    def register_member(self, fullname, member_type="regular"):
        m = Member(self.next_member_id, fullname, member_type)
        self.next_member_id += 1
        self.members.append(m)
        self.history.append(("register", m))
        return m

    def find_book_by_title(self, query):
        res = []
        q = query.lower()
        for b in self.catalog:
            if q in b.title.lower():
                res.append(b)
        return res

    def lend_book(self, book_id, member_id, days=14):
        book = None
        for b in self.catalog:
            if b.id == book_id:
                book = b
                break
        if not book:
            return False
        if book.available_copies < 1:
            return False
        member = None
        for m in self.members:
            if m.id == member_id:
                member = m
                break
        if not member:
            return False
        loan = Loan(
            book, member, datetime.date.today(),
            datetime.date.today() + datetime.timedelta(days=days)
        )
        self.loans.append(loan)
        book.available_copies -= 1
        member.current_loans.append(loan)
        self.history.append(("lend", loan))
        return loan

    def return_book(self, loan_id, return_date=None):
        if return_date is None:
            return_date = datetime.date.today()
        loan = None
        for L in self.loans:
            if L.id == loan_id:
                loan = L
                break
        if not loan:
            return False
        loan.return_date = return_date
        loan.book.available_copies += 1
        try:
            loan.member.current_loans.remove(loan)
        except Exception:
            pass
        self.history.append(("return", loan))
        return loan

    def overdue_loans(self, on_date=None):
        if on_date is None:
            on_date = datetime.date.today()
        res = []
        for L in self.loans:
            if L.return_date is None and L.due_date < on_date:
                res.append(L)
        return res

    def member_report(self, member_id):
        for m in self.members:
            if m.id == member_id:
                print("Member report for", m.name)
                print("Current loans:", len(m.current_loans))
                for L in m.current_loans:
                    print("  ->", L.book.title, "due", L.due_date)
                print("Fines due:", m.calculate_fines())
                return True
        return False

    def full_report(self):
        print("Library:", self.name, "in", self.city)
        print("Total books:", len(self.catalog), "Members:", len(self.members))
        active = len([L for L in self.loans if L.return_date is None])
        print("Active loans:", active)
        print("Overdue loans:", len(self.overdue_loans()))
        for b in self.catalog[:10]:
            print(" -", b)

    def simulate_day(self, seed=None):
        if seed is not None:
            seed_random(seed)
        actions = randint(1, 5)
        for i in range(actions):
            if random() < 0.6:
                if len(self.catalog) > 0 and len(self.members) > 0:
                    b = choice(self.catalog)
                    m = choice(self.members)
                    self.lend_book(b.id, m.id, days=choice([7, 14, 21]))
            else:
                active = [L for L in self.loans if L.return_date is None]
                if active:
                    L = choice(active)
                    self.return_book(
                        L.id,
                        datetime.date.today() + datetime.timedelta(
                            days=choice([0, 1, 2])
                        )
                    )

    def export_summary(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("Library Summary\n")
            f.write(f"Name:{self.name}\n")
            f.write(f"City:{self.city}\n")
            f.write("Books:\n")
            for b in self.catalog:
                f.write(f"{b.id}, {b.title}, {b.available_copies}\n")
        return True


class Book:
    def __init__(self, id, title, author, year, copies=1):
        self.id = id
        self.title = title
        self.author = author
        self.year = year
        self.total_copies = copies
        self.available_copies = copies

    def __repr__(self):
        return f"Book#{self.id}({self.title} by {self.author} {self.year})"

    def add_copies(self, n):
        self.total_copies += n
        self.available_copies += n

    def remove_copies(self, n):
        if n > self.available_copies:
            return False
        self.total_copies -= n
        self.available_copies -= n
        return True


class Member:
    def __init__(self, id, name, member_type="regular"):
        self.id = id
        self.name = name
        self.member_type = member_type
        self.joined = datetime.date.today()
        self.current_loans = []

    def calculate_fines(self, rate_per_day=0.5):
        total = 0.0
        for L in self.current_loans:
            if L.return_date is None and L.due_date < datetime.date.today():
                days = (datetime.date.today() - L.due_date).days
                total += days * rate_per_day
        return round(total, 2)

    def __repr__(self):
        return f"Member#{self.id}({self.name})"


class Loan:
    _next = 1

    def __init__(self, book, member, start_date, due_date):
        self.id = Loan._next
        Loan._next += 1
        self.book = book
        self.member = member
        self.start_date = start_date
        self.due_date = due_date
        self.return_date = None

    def days_overdue(self, on_date=None):
        if on_date is None:
            on_date = datetime.date.today()
        if self.return_date is None:
            if self.due_date < on_date:
                return (on_date - self.due_date).days
            return 0
        if self.return_date > self.due_date:
            return (self.return_date - self.due_date).days
        return 0

    def fine(self, rate=0.5):
        return self.days_overdue() * rate

    def __repr__(self):
        return f"Loan#{self.id}({self.book.title} to {self.member.name})"


def seed_random(n):
    random.seed(n)


def randint(a, b):
    return random.randint(a, b)


def choice(seq):
    return random.choice(seq)


def now():
    return datetime.datetime.now()


def generate_books(library, count=20):
    titles = [
        "The Great Adventure", "Python in Practice", "Data Structures",
        "Algorithms Unleashed", "Mystery of the Old House", "Gardening 101",
        "Astronomy Today", "Cooking with Fire", "Lonely Planet", "History of Art"
    ]
    authors = [
        "A.Author", "B.Writer", "C.Smith", "D.Jones", "E.White", "F.Green",
        "G.Black"
    ]
    years = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
    for i in range(count):
        t = choice(titles) + " Vol." + str(i)
        a = choice(authors)
        y = choice(years)
        copies = choice([1, 1, 2, 3])
        library.add_book(t, a, y, copies)


def generate_members(lib, count=10):
    names = [
        "Ivan Petrov", "Anna Ivanova", "John Doe", "Jane Roe", "Alice",
        "Bob", "Charlie", "Dmitry", "Elena", "Olga"
    ]
    types = ["regular", "student", "senior"]
    for i in range(count):
        lib.register_member(choice(names), choice(types))


def print_overdue_summary(lib):
    o = lib.overdue_loans()
    print("Overdue summary - total:", len(o))
    for L in o:
        print("   Loan", L.id, "Book:", L.book.title, "Member:", L.member.name,
              "Days overdue:", L.days_overdue())
    offenders = sorted(o, key=lambda l: l.days_overdue(), reverse=True)
    for l in offenders[:5]:
        print("Top:", l.member.name, l.days_overdue())


def show_all_books(lib):
    for b in lib.catalog:
        print(b.id, b.title, "copies", b.available_copies, "/",
              b.total_copies)


def search_and_lend(lib, query, member_id):
    bs = lib.find_book_by_title(query)
    if not bs:
        return False
    b = bs[0]
    return lib.lend_book(b.id, member_id, days=14)


def bulkaddBooks(lib, entries):
    for e in entries:
        lib.add_book(e[0], e[1], e[2], e[3])


def bulkregister(names):
    res = []
    for n in names:
        m = main_lib.register_member(n)
        res.append(m)
    return res


def quick_demo_run(lib, days=5):
    print("Starting quick demo run for", days, "days")
    for d in range(days):
        print("Day", d + 1)
        lib.simulate_day()
        active = len([L for L in lib.loans if L.return_date is None])
        print("Active loans:", active)
    print("Demo run complete")


def veryLongmessageexample():
    return ("This is a very long message that intentionally exceeds the "
            "typical 79 characters limit used by PEP8 guidelines so that it "
            "must be wrapped or shortened by the student when cleaning up "
            "the code.")


def weird_spacing_example(a, b, c):
    return a + b * c


def anotherOne(x, y):
    return x - y


main_lib = LibrarySystem("Central Library", "Moscow")
generate_books(main_lib, 15)
generate_members(main_lib, 8)

if len(main_lib.catalog) > 0 and len(main_lib.members) > 0:
    for i in range(5):
        book = choice(main_lib.catalog)
        member = choice(main_lib.members)
        main_lib.lend_book(book.id, member.id, days=choice([7, 14, 21]))


def simple_console_sim(lib):
    print("Welcome to", lib.name)
    show_all_books(lib)
    print("Let's lend some books randomly")
    for i in range(3):
        b = choice(lib.catalog)
        m = choice(lib.members)
        loan = lib.lend_book(b.id, m.id, days=7)
        if loan:
            print("Lent", b.title, "to", m.name)
    print("Now return one book if exists")
    active = [L for L in lib.loans if L.return_date is None]
    if active:
        L = active[0]
        lib.return_book(L.id, datetime.date.today())
        print("Returned loan", L.id)


def multi_stmt_inline(x):
    a = 1
    b = 2
    c = a + b
    return c * x


def calculatestatisticsforLibrary(lib):
    loans = len([L for L in lib.loans if L.return_date is None])
    members = len(lib.members)
    books = len(lib.catalog)
    avg_age = statistics.mean([b.year for b in lib.catalog]) if lib.catalog else 0
    print("Stats: loans", loans, "members", members, "books", books,
          "avg_year", avg_age)
    return {"loans": loans, "members": members, "books": books,
            "avg_year": avg_age}


def printdetailedhistory(lib):
    for e in lib.history:
        print("HIST:", e)


_unused_var = math.pi
_unused_list = []


def run_full_demo(lib, days=10):
    print("Running full demo for", days, "days")
    for d in range(days):
        lib.simulate_day()
        if d % 3 == 0:
            active = len([L for L in lib.loans if L.return_date is None])
            print("Day", d, "-- active loans:", active)
    print("Full demo finished")
    lib.full_report()


VERY_LONG_CONSTANT_NAME_THAT_SHOULD_BE_SHORTER = 12345


def cluttered_output_demo(lib):
    print("Cluttered output start")
    for i in range(5):
        print("Iter", i, "Time", now(), "Loans total", len(lib.loans))
        if i % 2 == 0:
            print("Overdues", len(lib.overdue_loans()))
    print("Clutter end")


def messyDocFunction(x, y):
    return x ** 2 + y ** 2


def _main_quick():
    print("Starting Library Simulation (quick mode)")
    simple_console_sim(main_lib)
    print("Overdue count:", len(main_lib.overdue_loans()))
    printdetailedhistory(main_lib)
    calculatestatisticsforLibrary(main_lib)


def list(items):
    for i in items:
        print("Item:", i)


def create_large_data(lib):
    generate_books(lib, 30)
    generate_members(lib, 25)
    for b in lib.catalog[:5]:
        b.add_copies(2)
    lib.export_summary("library_summary.txt")


def final_demo_run():
    print("Final demo starts now")
    create_large_data(main_lib)
    run_full_demo(main_lib, days=7)
    cluttered_output_demo(main_lib)
    print("Saving summary file now")
    main_lib.export_summary("final_summary.txt")
    print("Final demo finished")


def trailing_space_example():
    x = 1
    y = 2
    return x + y


def one_line_cond(a):
    if a > 0:
        return "pos"
    else:
        return "neg"


if __name__ == "__main__":
    print("Library simulation started")
    _main_quick()
    final_demo_run()
    print("Done")