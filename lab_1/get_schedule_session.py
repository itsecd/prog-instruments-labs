import csv

import requests
from bs4 import BeautifulSoup as BS


def find_schedule_session_url(num_group: str) -> str:
    """
    Finds the URL for the session schedule based on the group number.

    :param num_group: The group number to search for.
    :return: The URL for the session schedule for the specified group.
    """
    with open(f'AllGroupShedule/AllGroup_course_{num_group[1]}.csv',
              'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            if row[1] == num_group:
                result = f'{row[0][:20]}/session?groupId={row[0][29:]}'
                return result


def pars_schedule_session(num_group: str) -> str:
    """
    Parses the session schedule for a given group number.

    :param num_group: The group number for which to retrieve the session schedule.
    :return: A formatted string containing the session schedule for the specified group.
    """
    url = find_schedule_session_url(num_group)
    response = requests.get(url)
    list_session = []
    soup = BS(response.content, 'html.parser')
    for session_item in soup.find_all(class_='daily-session'):
        if not session_item.find(class_='caption-text meeting__type').contents[0] == ' Консультация':
            list_session.append((
                session_item.find(class_='h3-text').contents,
                session_item.find(class_='caption-text meeting__type').contents,
                session_item.find(class_='h3-text meeting__discipline').contents,
                session_item.find(class_='h2-text meeting__time').contents
            ))
    result = f'Расписание сессии для группы {num_group}:\n'
    for session in list_session:
        result += (f'---{session[1][0]}---\n\nДата и время: {session[0][0]} - '
                   f'{session[3][0]}\nПредмет: {session[2][0]}\n\n')
    return result
