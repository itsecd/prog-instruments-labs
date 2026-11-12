import math


def statement(invoice: dict, plays: dict) -> str:
    def format_usd(amount): return f"${amount:0,.2f}"

    def play_for(performance: dict) -> dict: return plays[performance['playID']]

    def amount_for(performance: dict) -> int:
        play = play_for(performance)
        match play['type']:
            case "tragedy":
                amount = 40000
                if performance['audience'] > 30: 
                    amount += 1000 * (performance['audience'] - 30)
            case "comedy":
                amount = 30000
                if performance['audience'] > 20:
                    amount += 10000 + 500 * (performance['audience'] - 20)
                amount += 300 * performance['audience']
            case _:
                raise ValueError(f'unknown type: {play["type"]}')
        return amount

    def volume_credits_for(performance: dict) -> int:
        result = max(performance['audience'] - 30, 0)
        match play_for(performance)['type']:
            case "comedy":
                # add extra credit for every ten comedy attendees
                result += math.floor(performance['audience'] / 5)
        return result
    
    def total_volume_credits() -> int:
        volume_credits = 0
        for perf in invoice['performances']:
            volume_credits += volume_credits_for(perf)
        return volume_credits

    def total_amount() -> int:
        total_amount = 0
        for perf in invoice['performances']:
            total_amount += amount_for(perf)
        return total_amount

    
    result = f'Statement for {invoice["customer"]}\n'
    for perf in invoice['performances']:
        result += f' {play_for(perf)["name"]}: {format_usd(amount_for(perf)/100)} ({perf["audience"]} seats)\n'

    result += f'Amount owed is {format_usd(total_amount()/100)}\n'
    result += f'You earned {total_volume_credits()} credits\n'
    return result
