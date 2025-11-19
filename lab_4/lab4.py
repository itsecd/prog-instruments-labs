from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException
from multiprocessing import Pool, freeze_support
from itertools import repeat
from fake_useragent import UserAgent
from random import uniform
from time import sleep


def random_pause():
    sleep(round(uniform(1.8, 2.1), 2))


def check_translated(lines: list[str], i: int, s: str) -> bool:
    """
    The method checks if the strings are translated
    Args:
        lines: All lines
        i: Current index of string
        s: Current string

    Returns: True if already translated
    """
    if s.startswith('msgid "'):
        # length (msgid ""\n) is 9
        # One-line - Simple string
        if len(s) > 9:
            try:
                next_str = lines[i + 1]
                if next_str.startswith('msgstr "') and len(next_str) > 10:
                    return True  # Simple translated string
            except IndexError:
                pass

        # More than one line - Complex string
        elif len(s) == 9:
            try:
                plus = 1
                while True:
                    next_str = lines[i + plus]
                    plus += 1
                    # length (msgstr ""\n) is 10
                    if next_str.startswith('msgstr "'):
                        next_str2 = lines[i + plus]
                        if (len(next_str) > 10) or (next_str2.startswith('"') and len(next_str2) > 2):
                            # Complex translated string
                            return True
                        return False
            except IndexError:
                pass

        # Isn't translated
        return False


def translate(dr: webdriver.Chrome, text: str, last_trans: str, retry: int) -> str:
    """
    This method translates received strings
    Args:
        dr: Driver
        text: Text to translate
        last_trans: Last translate
        retry: Attempts to translate

    Returns: Translated text
    """
    substitution, substitution2, variables, unf, service_f = "{%s}", "{+}", list(), False, False

    # Check python unnamed-format
    if '%s' in text:
        unf = True

    # Save python named-format
    count_format = text.count('%(')
    for i in range(count_format):
        f_pos = text.find('%(')
        s_pos = text[f_pos:].find(')s')
        named_format = text[f_pos:][:s_pos + 2]  # + ')s'
        variables.append(named_format)
        text = text.replace(named_format, substitution % (i + 1,), 1)

    # Prepare string to work
    if r'\"' in text:
        service_f = True
        text = text.replace(r'\"', '"')

    # Save html classes
    classes = list()
    count_classes = text.count('class="')  # Only this entry: class=""
    for i in range(count_classes):
        f_pos = text.find('class="')
        s_pos = text[f_pos + 7:].find('"')  # + 7 == len('class="')
        cls = f'class="{text[f_pos + 7:][:s_pos]}"'
        text = text.replace(cls, substitution2, 1)
        classes.append(cls)

    # Working with translator fields
    # textarea.er8xn -> input field
    # span.ryNqvb -> output text
    def take_text() -> str:
        """
        Takes text from the output field and formats
        NoSuchElementException thrown if the page didn't load correctly
        """
        random_pause()
        try:
            trans_text = dr.find_elements(by=By.CSS_SELECTOR, value='span.ryNqvb')
        except NoSuchElementException:
            dr.refresh()
            trans_text = dr.find_elements(by=By.CSS_SELECTOR, value='span.ryNqvb')

        if not trans_text:  # Multiple translations
            # For example, in French there can be 2 translations of one word.
            trans_text = dr.find_elements(by=By.CSS_SELECTOR, value='span.HwtZe')[-1:]
        trans_text = [sentence.text for sentence in trans_text]
        trans_text = ''.join(trans_text)
        return trans_text

    try:
        dr.find_element(by=By.CSS_SELECTOR, value='textarea.er8xn').clear()
        dr.find_element(by=By.CSS_SELECTOR, value='textarea.er8xn').send_keys(text)
        trans = take_text()

        if trans == last_trans:  # If GT is late
            trans = take_text()

    except Exception as ex:
        random_pause()
        if retry:
            print(f'[!] FAIL -> {text} | retry={retry} ({ex})')
            retry -= 1
            return translate(dr=dr, text=text, last_trans=last_trans, retry=retry)
        else:
            print(f'[!] No attempts left for -> {text}')
            return ""

    # Fix python unnamed-format
    if unf:
        trans = trans.replace('%S', '%s')

    # Return python named-format
    for i in range(len(variables)):
        trans = trans.replace(substitution % (i + 1,), variables[i], 1)

    # Return html classes
    for i in range(len(classes)):
        trans = trans.replace(substitution2, classes[i], 1)

    # Fix python service str
    if service_f:
        trans = trans.replace('"', r'\"')

    print(f'[+] {text} - {trans}')
    return trans


def translator(
        code: str,
        driver_path: str,
        locale_path: str,
        headless: bool,
        lang_interface: str,
        from_lang: str,
        retry: int
):
    """
    Initialization of all variables. And the translation.
    Args:
        code: Language code
        driver_path: Path to chromedriver
        locale_path: Path to locale folder
        headless: Windowless mode
        lang_interface: Language (code) in which GT will be opened
        from_lang: Language code from which the translation will be carried out
        retry: Attempts to translate

    Returns: None
    """
    url = 'https://translate.google.com/?hl=%(lang_interface)s&sl=%(from_lang)s&tl=%(to_lang)s&op=translate'
    user_agent = UserAgent(verify_ssl=False)
    s = Service(executable_path=driver_path)
    options = webdriver.ChromeOptions()
    options.add_argument(argument=f'user-agent={user_agent.random}')
    options.add_argument(argument='--disable-blink-features=AutomationControlled')
    if headless:
        options.add_argument(argument='--headless')
    dr = webdriver.Chrome(service=s, options=options)
    solved_url = url % {'lang_interface': lang_interface,
                        'from_lang': from_lang,
                        'to_lang': code}
    dr.get(solved_url)
    modified = False

    # Read file
    path_file = f'{locale_path}/{code}/LC_MESSAGES/django.po'
    try:
        with open(path_file, 'r', encoding='UTF-8') as file:
            print(f'{path_file} - opened!')
            text, found, translated, next_complex, save_complex, last_trans = '', False, None, False, False, None
            to_translate = list()
            lines = file.readlines()
    except FileNotFoundError:
        print(f"[!] FAIL {path_file} doesn't exists")
        return

    # i - string index
    # s - the string
    for i, s in enumerate(lines):
        # Checking for already translated
        if check_translated(lines=lines, i=i, s=s):
            text += s
            continue

        if s.startswith('msgid "'):  # Define text for translation
            string_text = s[s.find('"') + 1:s.rfind('"')].strip()
            if string_text:  # Simple translation
                translated, found = translate(dr=dr, text=string_text, last_trans=last_trans, retry=retry), True
                last_trans = translated
            else:  # Complex trans
                if lines[i + 1].startswith('"'):  # It's complex text
                    next_complex = True
            text += s

        elif s.startswith('"') and next_complex and len(s) > 2:  # Complex text
            to_translate.append(s[s.find('"') + 1:s.rfind('"')])
            text += s
            # Check next line
            try:
                if lines[i + 1].startswith('msgstr "'):
                    next_complex, save_complex, found = False, True, True
            except IndexError:
                print(f'[!] SyntaxError at {i + 1} line in {path_file}')

        elif s.startswith('msgstr "') and found:  # Write translated text
            if save_complex:  # Write complex
                solved = translate(dr=dr, text=' '.join(to_translate), last_trans=last_trans, retry=retry)
                last_trans = solved
                text += ('msgstr ""\n"' + solved + '"\n')
                save_complex = False
                to_translate.clear()
            else:  # Save simple
                text += f'msgstr "{translated}"\n'
            modified = True
            translated, found = None, False
        else:  # Just text
            text += s

    # Dump
    if modified:
        with open(path_file, 'w', encoding='UTF-8') as file:
            file.write(text)
            print(path_file, '- Saved!')
    else:
        print(path_file, '- Without changes!')


def manager(codes: list,
            driver_path: str,
            locale_path: str,
            headless: bool = True,
            multi: bool = False,
            multi_max: int = 10,
            lang_interface: str = 'en',
            from_lang: str = 'en',
            retry: int = 3
            ):
    """
    The manager handles the multiprocessing mode
    Args:
        codes: Language codes
        driver_path: Path to chromedriver
        locale_path: Path to locale folder
        headless: Windowless Mode
        multi: Multiprocessor mode
        multi_max: Maximum number of processes
        lang_interface: Language (code) in which GT will be opened
        from_lang: Language code from which the translation will be carried out
        retry: Attempts to translate

    Returns: None
    """
    if multi:
        freeze_support()
        variables_copy = codes.copy()
        while variables_copy:
            langs = variables_copy[:multi_max]
            variables_copy = variables_copy[multi_max:]
            with Pool(processes=len(langs)) as pool:
                start = pool.starmap(translator, zip(langs,
                                                     repeat(driver_path),
                                                     repeat(locale_path),
                                                     repeat(headless),
                                                     repeat(lang_interface),
                                                     repeat(from_lang),
                                                     repeat(retry)))
    else:
        for code in codes:
            translator(code, driver_path, locale_path, headless, lang_interface, from_lang, retry)


if __name__ == '__main__':
    manager(
        codes=['de', 'fr', 'ja', 'tr', 'ru', 'uk'],
        driver_path='/DJTranslator/chromedriver',
        locale_path='/DJAuth/locale',
        multi=True,
        lang_interface='ru',
    )
