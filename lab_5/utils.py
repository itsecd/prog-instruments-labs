def format_currency(amount: float) -> str:
    if amount < 0:
        raise ValueError("Сумма не может быть отрицательной")
    return f"{amount:.2f} руб."
