from expense_report import ExpenseReport, Expense, ExpenseType

if __name__ == '__main__':
    expense = Expense()
    expense.type = ExpenseType.DINNER
    expense.amount = 7500
    ExpenseReport().print_report([expense])