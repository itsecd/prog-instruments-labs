#!/usr/bin/env python3
"""
Expense Tracker (CLI)
---------------------
A tiny, dependency-free CLI to log and summarize expenses.
Stores data in a CSV file: date,amount,category,note

Examples:
    python expense_tracker.py add --amount 12.5 --category Food --note "Lunch"
    python expense_tracker.py list
    python expense_tracker.py summary --by category
    python expense_tracker.py summary --by month
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

DATA_FILE = Path(__file__).with_name("expenses.csv")
logger = logging.getLogger(__name__)


@dataclass
class Expense:
    date: str  # ISO format YYYY-MM-DD
    amount: float
    category: str
    note: str = ""

    @staticmethod
    def from_args(args: argparse.Namespace) -> Expense:
        # Parse/validate date
        if args.date:
            try:
                d = datetime.strptime(args.date, "%Y-%m-%d").date()
            except ValueError as e:
                raise SystemExit(f"Invalid --date format (use YYYY-MM-DD): {e}")
        else:
            d = date.today()
        if args.amount <= 0:
            raise SystemExit("--amount must be positive.")
        return Expense(
            date=d.isoformat(),
            amount=float(args.amount),
            category=args.category.strip(),
            note=(args.note or "").strip(),
        )


def ensure_file_exists() -> None:
    if not DATA_FILE.exists():
        with DATA_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["date", "amount", "category", "note"]
            )
            writer.writeheader()


def read_expenses() -> list[Expense]:
    ensure_file_exists()
    items: list[Expense] = []
    with DATA_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                items.append(
                    Expense(
                        date=row["date"],
                        amount=float(row["amount"]),
                        category=row["category"],
                        note=row.get("note", ""),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                logger.warning("Skipping malformed CSV row: %s", error)
                continue
    return items


def write_expense(expense: Expense) -> None:
    ensure_file_exists()
    with DATA_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "amount", "category", "note"])
        writer.writerow(asdict(expense))


def cmd_add(args: argparse.Namespace) -> None:
    exp = Expense.from_args(args)
    write_expense(exp)
    print(f"Added: {exp.date}  ${exp.amount:.2f}  {exp.category}  {exp.note}")


def cmd_list(args: argparse.Namespace) -> None:
    items = read_expenses()
    if not items:
        print("No expenses yet. Use 'add' to create one.")
        return
    print(f"{'DATE':<12} {'AMOUNT':>10}  {'CATEGORY':<15}  NOTE")
    print("-" * 60)
    for e in items:
        print(f"{e.date:<12} {e.amount:>10.2f}  {e.category:<15}  {e.note}")


def summarize_by_category(items: Iterable[Expense]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for e in items:
        totals[e.category] += e.amount
    return dict(sorted(totals.items(), key=lambda kv: kv[0].lower()))


def summarize_by_month(items: Iterable[Expense]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for e in items:
        try:
            y, m, _ = e.date.split("-")
            key = f"{y}-{m}"
        except ValueError:
            key = "unknown"
        totals[key] += e.amount
    # Sort by YYYY-MM
    return dict(sorted(totals.items(), key=lambda kv: kv[0]))


def cmd_summary(args: argparse.Namespace) -> None:
    items = read_expenses()
    if not items:
        print("No expenses yet. Use 'add' to create one.")
        return
    if args.by == "category":
        totals = summarize_by_category(items)
        print("Summary by category")
    else:
        totals = summarize_by_month(items)
        print("Summary by month (YYYY-MM)")
    print("-" * 40)
    grand = 0.0
    for k, v in totals.items():
        print(f"{k:<15} ${v:>10.2f}")
        grand += v
    print("-" * 40)
    print(f"{'TOTAL':<15} ${grand:>10.2f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Expense Tracker CLI")
    sub = p.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add a new expense")
    p_add.add_argument("--amount", type=float, required=True, help="Amount spent")
    p_add.add_argument(
        "--category", type=str, required=True, help="Category (e.g., Food, Rent)"
    )
    p_add.add_argument("--note", type=str, help="Optional note")
    p_add.add_argument("--date", type=str, help="YYYY-MM-DD (defaults to today)")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="List all expenses")
    p_list.set_defaults(func=cmd_list)

    p_sum = sub.add_parser("summary", help="Summarize expenses")
    p_sum.add_argument("--by", choices=["category", "month"], default="category")
    p_sum.set_defaults(func=cmd_summary)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
