import math


def amount_for(play: dict, performance: dict) -> int:
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


def volume_credits_for(play: dict, performance: dict) -> int:
    result = max(performance['audience'] - 30, 0)
    match play['type']:
        case "comedy":
            # add extra credit for every ten comedy attendees
            result += math.floor(performance['audience'] / 5)
    return result


def statement(invoice: dict, plays: dict) -> str:
    total_amount = 0
    volume_credits = 0
    result = f'Statement for {invoice["customer"]}\n'

    def format_as_dollars(amount):
        return f"${amount:0,.2f}"

    for perf in invoice['performances']:
        play = plays[perf['playID']]
        this_amount = amount_for(play, perf)
        # add volume credits
        volume_credits += volume_credits_for(play, perf)
        total_amount += this_amount
        # print line for this order
        result += f' {play["name"]}: {format_as_dollars(this_amount/100)} ({perf["audience"]} seats)\n'

    result += f'Amount owed is {format_as_dollars(total_amount/100)}\n'
    result += f'You earned {volume_credits} credits\n'
    return result
