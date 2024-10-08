import csv
import os

import requests
from bs4 import BeautifulSoup as BS


def pars_all_group():
    """
    Parses all group schedules from the university website and saves them
    in CSV files organized by course number.

    Creates a directory named 'AllGroupShedule' if it does not exist.
    Retrieves group URLs and names for courses 1 to 5 and stores them in
    respective CSV files.
    """
    os.makedirs('AllGroupShedule', exist_ok=True)
    for num_course in range(1, 6):
        url = f'https://ssau.ru/rasp/faculty/492430598?course={num_course}'
        response = requests.get(url)
        soup = BS(response.content, 'html.parser')
        list_url = []
        list_group = []
        for item in soup.find_all('a', class_='btn-text group-catalog__group'):
            list_url.append(item.get('href'))
            span_item = item.find('span')
            list_group.append(span_item.contents[0])

        for i in range(len(list_url)):
            with open(f'AllGroupShedule/AllGroup_course_{num_course}.csv',
                      'a', newline='', encoding='utf-8') as file:
                printer = csv.writer(file, delimiter=';')
                printer.writerow([
                    'https://ssau.ru' + str(list_url[i]),
                    str(list_group[i])
                ])


def find_schedule_url(num_group: str, selected_week: str, selected_weekday: str) -> str:
    """
    Finds the schedule URL for a specific group, week, and weekday.

    :param num_group: The group number to search for.
    :param selected_week: The week number for which to find the schedule.
    :param selected_weekday: The day of the week for which to find the schedule.
    :return: The constructed URL for the schedule.
    """
    with open(f'AllGroupShedule/AllGroup_course_{num_group[1]}.csv',
              'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            if row[1] == num_group:
                result = (f'{row[0]}&selectedWeek='
                          f'{selected_week}&selectedWeekday={selected_weekday}')
                return result


def pars_shedule(url: str) -> str:
    """
    Parses the schedule from the given URL.

    :param url: The URL of the schedule to parse.
    :return: A formatted string containing the schedule information.
    """
    result = ''
    list_lessons = []
    list_lessons_time = [
        '8:00 - 9:35',
        '9:45 - 11:20',
        '11:30 - 13:05',
        '13:30 - 15:05',
        '15:15 - 16:50',
        '17:00 - 18:35'
    ]
    list_lessons_type = []
    response = requests.get(url)
    soup = BS(response.content, 'html.parser')
    current_date = soup.find(class_='week-nav-current_date')
    result += f'Расписание на {current_date.contents[0]}\n\n'

    for item in soup.find_all(class_='schedule__item schedule__item_show'):
        i = 1
        last_num_of_lessons = len(list_lessons)
        while i != 5:
            try:
                name_lesson = item.find(
                    class_=f'body-text schedule__discipline lesson-color lesson-color-type-{i}')
                list_lessons.append(str(name_lesson.contents[0]))
                if last_num_of_lessons < len(list_lessons):
                    if i == 1:
                        list_lessons_type.append('(ЛЕКЦИЯ)')
                    if i == 2:
                        list_lessons_type.append('(ЛАБОРАТОРНАЯ)')
                    if i == 3:
                        list_lessons_type.append('(ПРАКТИКА)')
                    if i == 4:
                        list_lessons_type.append('(ДРУГОЕ)')
            except:
                pass
            i += 1
        if last_num_of_lessons == len(list_lessons):
            list_lessons.append('None')
            list_lessons_type.append('(None)')

    for i in range(len(list_lessons)):
        result += (f'{list_lessons_time[i]}\t{list_lessons[i]}'
                   f'\t{list_lessons_type[i]}\n')

    return result
