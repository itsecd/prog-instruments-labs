#! python3.6
# FB_Bot.py - Scraper for Facebook pages that sends posts to Telegram channels
# version is 20181001
# if you want to say anything go to @Udinanon
# on Telegram or check my email here on GitHub
# DISTRIBUTED UNDER GNU LGPL v3 or latest
# THE AUTHOR OF THE SCRIPT DOES NOT AUTHORIZE MILITARY
# USE OF HIS WORK OR USAGE IN ANY MILITARY-REALTED
# ENVIROMENT WITHOUT HIS EXPLICIT CONSENT

# TO DO LIST:
# better comment the code //getting better
# reorder the code and make it more readable //it's getting better
# handle HD photos
# handle continue reading in very long posts

import argparse
import configparser
import csv
import cgi
import json
import logging
import re
import time
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from requests_html import HTMLSession
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

TG_BASE = "https://api.telegram.org/bot{}/"
# This is used to check the length of messages later
TAG_REGEX = re.compile(r'<[^>]+>')
TIME_REGEX = re.compile(r"(#)(1[0-9]{9})(#)")
USER_AGENT = {
    "User-Agent": (
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
    )
}
FIREFOX_OPTIONS = Options()
FIREFOX_OPTIONS.headless = True


# basic stuff

def create_session():
    session = requests.Session()
    retry = Retry(total=10, connect=3, redirect=5, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def get_url(url):  # get webpages as general files
    with create_session() as session:
        response = session.get(url, headers=USER_AGENT)
    logging.debug("GET URL:" + url + "\nRESPONSE:" + response.reason)
    return response.content


def get_date():
    date = time.strftime("%Y-%m-%d+%H.%M")
    return date


def get_day(form="%y%m%d"):
    day = time.strftime(str(form))
    return day


# id is used to determine the age of a post and to avoid duplicates


"""
The CSV file is just a list of the FB pages to be scraped.
The first line is ignored as it is supposed to be human readable.

It is structured as follows:
    [0] A human readable name, not used by the script
    [1] The name that is sent as the title of the post,
        usually a null string for single page channels and
        the page's name for multipage channels
    [2] The URL of the posts section of the page
    [3] The UNIX time of the last read post,
        set to 0 if it's the first time so every post is sent
    [4] Telegram channel ID on which the posts are supposed to go
"""


def update_csv(pages, input_file):  # write new data to csv of Facebook Pages
    with open(input_file, "w", newline='', encoding='utf_8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow(
            ("HUMAN READABLE", "NAME", "URL", "LAST_TIME", "TOKEN", "ID")
        )
        for row in pages:
            writer.writerow(row)


def get_post_time(post):  # find the UNIX time of the post
    time_area = post.select_one("abbr._5ptz")
    if time_area is None:
        return 0
    post_time = time_area["data-utime"]
    return int(post_time)


def get_mobile_url(url):
    return url[:8:] + "m" + url[11::]


def get_page_name(url):
    session = HTMLSession()
    url = get_mobile_url(url)
    r = session.get(url)
    name = r.html.find("title", first=True)
    return name.text.replace(" - Post | Facebook", "")


def add_video_link(post):  # add video link to top of the post's text
    text = "<a href='" + post["video"] + "'>VIDEO</a> \n" + post["text"]
    return text


def add_link(post):  # add link to the top of the post's text
    text = "<a href='" + post["link"] + "'>LINK</a> \n" + post["text"]
    return text


# add link to the Facebook post at the bottom of the post
def add_link_to_post(post):
    post["link2post"] = handle_link_to_post(post)
    text = (
        f"{post['text']}\n<a href='{post['link2post']}'>POST</a>"
    )
    return text


def add_page_name(post):  # add page name in bold to the top of the post
    text = "<b>" + str(post["page_name"]) + "</b>\n" + post["text"]
    return text


# used to check if the shown message will be <200 chars in Telegram
def remove_tags(text):
    return TAG_REGEX.sub('', text)


def config_parser(ini_file):
    config = configparser.ConfigParser(interpolation=None)
    config.read(ini_file, encoding="utf-8")
    return config


def configure_logging(log_config):
    numeric_level = getattr(logging, log_config["debug_level"].upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid log level in configuration file:"
            f" {log_config['debug_level']}"
        )
    log_config["log_file"] = (
        f"{log_config['log_file_name']}"
        f"{get_day(log_config['date_structure'])}.log"
    )
    logging.basicConfig(filename=log_config["log_file"], level=numeric_level)


# description of the program and command line arguments
def argument_parser():
    parser = argparse.ArgumentParser(
        description="Scraper for Facebook pages that "
                    "sends posts to telegram channels"
    )
    parser.add_argument(
        "-ini_file",
        dest="ini_file",
        help="INI file from which all settings are read; "
             "defaults to ./FB_Bot.ini",
        default="./FB_Bot.ini"
    )
    parser.add_argument(
        "-D",
        dest="DEBUG_MODE",
        help="Forces DEBUG level logging, ignoring settings in ./FB_Bot.ini",
        action="store_true"
    )
    return vars(parser.parse_args())


# telegram relaying

def send_post(post):  # for text posts
    if len(remove_tags(
            # Recursively split text into 4000-character messages
            post["text"])) > 4000:
        print("LOONG MESSAGE")
        print("LENGHT: " + str(len(post["text"])))
        print(type(post))
        print(str(post))
        post1 = post.copy()
        post2 = post.copy()
        post1["text"] = post["text"][0:4000]
        print("PART 1 LENGHT: " + str(len(post1["text"])))
        post2["text"] = post["text"][4000:]
        print("PART 2 LENGHT: " + str(len(post2["text"])))
        send_post(post1)
        r = send_post(post2)
        return r
    url = TG_BASE.format(str(post["BOT"])) + "sendMessage"
    data = {
        "chat_id": post["channel_id"],
        "text": post["text"],
        "parse_mode": "html",
        "disable_web_page_preview": post.get("no_link")
    }
    r = requests.get(url, params=data, headers=USER_AGENT)
    log_message = ("SENDING POST, RESPONSE:" + r.reason
                   + " STATUS CODE:" + str(r.status_code))
    logging.info(log_message)
    if r.status_code != 200:
        logging.critical("THERE WAS AN ERROR IN A REQUEST IN send_post")
        logging.critical("URL: " + str(r.url))
        logging.critical("DATA: " + str(data))
        logging.critical("REASON: " + str(r.reason))
        logging.critical("RESPONSE: " + str(r))
    return r.json()["result"]["message_id"]


# download and upload photos, used as backup when sending the URL doesn't work
def send_photo_multipart(post):
    with open("temp.png", "wb") as file:
        file.write(requests.get(post["photo"]).content)
    photo = {"photo": open("temp.png", "rb")}
    url = TG_BASE.format(str(post["BOT"])) + "sendPhoto"
    data = {
        "chat_id": post["channel_id"],
        "caption": post["text"],
        "parse_mode": "html",
        "reply_to_message_id": post.get("reply_id")
    }
    r = requests.post(url, data=data, files=photo, headers=USER_AGENT)
    if r.status_code != 200:
        logging.critical("THERE WAS A N ERROR IN A REQUEST "
                         "IN send_photo_multipart")
        logging.critical("URL: " + str(r.url))
        logging.critical("DATA: " + str(data))
        logging.critical("REASON: " + str(r.reason))
        logging.critical("RESPONSE: " + str(r))
    return r


def send_photo(post):  # send photo via URL, with the text if <200
    # if the text is too long it isn't shown correctly by Telegram
    if len(remove_tags(post["text"])) > 195:
        # so it's sent first and then the photo as a reply to it
        post["reply_id"] = send_post(post)
        post["text"] = ""
    url = TG_BASE.format(str(post["BOT"])) + "sendPhoto"
    data = {
        "chat_id": post["channel_id"],
        "photo": post["photo"],
        "caption": post["text"],
        "parse_mode": "html",
        "reply_to_message_id": post.get("reply_id")
    }
    r = requests.get(url, params=data, headers=USER_AGENT)
    log_message = ("SENDING PHOTO, RESPONSE:" + r.reason
                   + " STATUS CODE:" + str(r.status_code))
    logging.info(log_message)
    # when Facebook links don't work
    if (
            r.status_code == 400
            and (
            r.json()["description"] == "Bad Request: wrong "
                                       "file identifier/HTTP URL specified"
            or r.json()["description"] == "Bad Request: failed "
                                          "to get HTTP URL content"
            )
    ):
        logging.warning("URL: %s", r.url)
        logging.warning("THERE WAS A PROBLEM IN THE REQUEST,"
                        " TRYING TO MULTIPART IT")
        logging.warning("DATA: %s", data)
        r = send_photo_multipart(post)
    elif r.status_code != 200:
        logging.critical("THERE WAS AN ERROR IN A REQUEST IN send_photo")
        logging.critical("URL: " + str(r.url))
        logging.critical("DATA: " + str(data))
        logging.critical("REASON: " + str(r.reason))
        logging.critical("RESPONSE: " + str(r.json()))
    return r.json()["result"]["message_id"]


def send_photos(post):  # used to send multiple photos in a chain of replies
    photos = post["photos"]
    post["photo"] = photos[0]
    reply_id = send_photo(post)
    photos = photos[1::]
    for photo in photos:
        post["text"] = ""
        post["photo"] = photo
        post["reply_id"] = reply_id
        reply_id = send_photo(post)


# content handling
# jesus facebook is fuc*ing awful


def handle_text(post):
    # here it's handling the text
    text_area = post["HTML"].select_one("div._5pbx.userContent")
    if text_area:  # here it's detected if it is there
        # hiding elements for long posts are detected
        useless_texts = text_area.find_all(class_="text_exposed_hide")
        for junk in useless_texts:
            junk.decompose()  # and eleminate

        strings = text_area.find_all(string=re.compile("[<>&]"))
        for string in strings:
            # here link are recompiled as text so they can be read later and
            # can be understood by Telegram
            string.replace_with(str(cgi.escape(string)))
        profile_links = text_area.find_all("a", class_="profileLink")
        for profile in profile_links:
            if profile.string is not None:
                # same thing for Facebook Profile Links
                profile.string.replace_with(str(profile))
        # the text is then extraced from the HTML code
        text = text_area.get_text()
        return str(post["text"] + str(text))
    else:
        return post["text"]


def handle_shares(post):
    try:
        share_area = post["HTML"].select_one("span._1nb_.fwn")
        shared_page_link, shared_page = (share_area.a["href"],
                                         share_area.a.string)
        link = ("\U0001F4E4 <a href='" + str(shared_page_link) + "'>"
                + str(shared_page) + "</a>\n")
        strings = share_area.next_sibling.next_sibling.find_all(
            string=re.compile("[<>&]")
        )
        for string in strings:
            # here link are recompiled as text so they can be read later and
            # can be understood by Telegram
            string.replace_with(str(cgi.escape(string)))
        text = str(share_area.next_sibling.next_sibling.get_text())
        text = str(link) + str(text) + "\n \n"
        return str(text)
    except Exception:
        return ""


def find_photo(post):
    # the area of the posts that handles a single photo
    # is identified with the unique class
    photo_area = post["HTML"].find("div", class_="_5cq3")
    if photo_area:  # here it's checked if it's really there
        photo = photo_area.find("img")  # the <img> tag is extrapolated
        # the link to the photo at the origin is extrapolated from the tag
        photo_link = photo["src"]
        return photo_link
    else:  # handling in case there is another type of photo in this post
        photo_area = post["HTML"].find("div", class_="_517g")
        if photo_area:  # here it's cheked if it's really there
            photo = photo_area.find("img")  # the <img> tag is extrapolated
            # the link to the photo at the origin is extrapolated from the tag
            photo_link = photo["src"]
            return photo_link
        else:  # handling in case there is no photo in this post
            return None


# basically the same as find_photo
# but with different tags for multiple photo posts
def find_photos(post):
    photos = []
    multi_photo_area = post["HTML"].find("div", class_="_2a2q")
    if multi_photo_area:
        photo_areas = multi_photo_area.find_all("a")
        for photo_area in photo_areas:
            try:
                photo_link = photo_area["data-ploi"]
            except KeyError:
                return None
            photos.append(photo_link)
        return photos
    else:
        return None


def parsing_link(query, fb_link):  # used to get around Facebook's secure link
    try:
        if query["u"] != "":
            link = str(query["u"][0])
            return link
    except KeyError:
        link = str(fb_link)
        return link


# used to parse a link out of Facebook's secure logout
def link_parse(fb_link):
    parsed_FB_link = urlparse(fb_link)
    query = parse_qs(parsed_FB_link.query)
    return parsing_link(query, fb_link)


# used to find the main link in link posts
def find_link(post):
    # this is for majority of link posts
    link_area = post["HTML"].find("a", class_="_52c6")
    if link_area:
        fb_link = link_area["href"]
        link = link_parse(fb_link)
        return link
    else:
        # for Youtube, twitch and other video links detected by FB
        link_area = post["HTML"].find("div", class_="mbs _6m6 _2cnj _5s6c")
        if link_area:
            # facebook hides the shared link with it's own "secure logout"
            fb_link = link_area.a["href"]
            link = link_parse(fb_link)
            return link
        else:
            return None


def has_video(post):  # to detect of a post has a Facebook video
    split_link2post = list(post["link2post"].split('/'))
    try:
        if split_link2post[4] == "videos":
            return True
    except Exception:
        return False


def find_video(post):
    # facebook mobile has plain link to the videos on their servers so
    # i can strip them and use those directly
    mobile_URL = get_mobile_url(post["link2post"])
    soup = BeautifulSoup(get_url(mobile_URL), "html.parser")
    video_areas = soup.find_all("a", target="_blank")
    if len(video_areas) != 0 and video_areas is not None:
        for video_area in video_areas:
            try:
                if (video_area.contents is not None
                        and video_area.contents[0].name != "span"):
                    video_link_dict = parse_qs(video_area["href"])
                    if "/video_redirect/?src" in video_link_dict:
                        video_link = video_link_dict["/video_redirect/?src"]
                    elif "https://lm.facebook.com/l.php?u" in video_link_dict:
                        video_link = video_link_dict[
                            "https://lm.facebook.com/l.php?u"
                        ]
                    return video_link[0]
            except IndexError:
                logging.debug("VIDEO AREA ERROR, SKIPPING")
                pass
            continue
    else:  # they seem to have changed how it works
        video_areas = soup.find("div", {"data-sigil": "inlineVideo"})
        if video_areas != [] and video_areas is not None:
            data = video_areas["data-store"]
            # yes it's a JSON dictionary inside an HTML tag,
            # I hate this timeline
            data_dict = json.loads(data)
            return data_dict["src"]
    # lives do have /videos/ in the URL
    # but don't have a video area that I can use,
    # to catch this possibility it returns -1
    return -1


def handle_link_to_post(post):  # to generate the link to the Facebook post
    link2post_area = post["HTML"].find("span", class_="fsm fwn fcg")
    try:
        link2post = "https://www.facebook.com" + link2post_area.a["href"]
        logging.info("HANDLING: " + str(link2post))
        return link2post
    except Exception:
        return ""


# used to detect and handle the different kinds of posts and contents
def content(post):
    post["text"] = handle_shares(post)
    post["text"] = handle_text(post)
    post["text"] = add_link_to_post(post)
    logging.debug("Basic text handled!")
    if find_photo(post):
        post["photo"] = find_photo(post)
        post["first_photo"] = True
        post["no_link"] = True
        post["text"] = add_page_name(post)
        logging.debug("Prepared the photo post")
        send_photo(post)
    elif find_link(post):
        post["link"] = find_link(post)
        post["text"] = add_link(post)
        post["text"] = add_page_name(post)
        logging.debug("prepared the link post")
        send_post(post)
    elif find_photos(post):
        post["photos"] = find_photos(post)
        post["no_link"] = True
        post["text"] = add_page_name(post)
        logging.debug("Prepared the multiple photo post")
        send_photos(post)
    elif has_video(post):
        post["video"] = find_video(post)
        post["no_link"] = True
        # if it's -1 then there was no video link and
        # so it is send like a normal text post
        if post["video"] != -1:
            post["text"] = add_video_link(post)
            post["no_link"] = False
            logging.debug("Found the video")
        post["text"] = add_page_name(post)
        logging.debug("Posting the video")
        send_post(post)
    else:
        post["no_link"] = True
        post["text"] = add_page_name(post)
        logging.debug("Sending the post")
        send_post(post)


# here it's checked of there are new posts
def new_posts_handling(posts, last_time, bot, channel_id, page_name):
    logging.debug("Last valid time: " + str(last_time))
    times = [int(last_time)]
    for element in posts:
        post_time = get_post_time(element)  # the Unix time is gathered
        if int(post_time) > int(last_time):
            post = {}
            post["HTML"] = element
            log_message = (
                "New post with post_time: " + str(post_time)
                + " for " + str(page_name)
            )
            logging.debug(log_message)
            post["BOT"], post["channel_id"], post["page_name"] = (bot,
                                                                  channel_id,
                                                                  page_name,
                                                                  )
            content(post)  # the post is handled
            # the new post time is added to the list
            times.append(int(post_time))
            logging.debug("Appended new post time: " + str(post_time))
    return max(times)  # the new top post time is returned


# A PATCH FOR PYPETEER CRASHING

def patch_pyppeteer():
    import pyppeteer.connection
    original_method = pyppeteer.connection.websockets.client.connect

    def new_method(*args, **kwargs):
        kwargs['ping_interval'] = None
        kwargs['ping_timeout'] = 40
        return original_method(*args, **kwargs)

    pyppeteer.connection.websockets.client.connect = new_method


# return None


def generate_soup(url):  # now we use RequestsHTML,
    # which can compile Javascript and
    # handes session errors better
    # with HTMLSession() as session:
    # retry = Retry(total=10, connect=3, redirect=5,
    # backoff_factor=0.5)  # i'm not even sure what it does
    # but helps with facebook stoppoing the bot
    # adapter = HTTPAdapter(max_retries=retry)
    # session.mount('http://', adapter)
    # session.mount('https://', adapter)
    # it's one of the reasons why I'm using RequestsHTML
    # r = session.get(URL)
    # patch_pyppeteer()
    # r.html.render()  # here the Javascript is rendered in a mock browser
    # body = r.html.find("body",
    # first=True).html
    # and the body of the page is extracetd to be prcessed by BeautifulSoup
    driver = webdriver.Firefox(options=FIREFOX_OPTIONS)
    driver.get(url)
    code = str(driver.page_source)
    driver.quit()
    return BeautifulSoup(code, "html.parser")


def gather_data(input_file):  # pages are loaded for the input file
    pages = []
    try:
        with open(input_file, "r", newline='', encoding='utf_8') as file:
            next(file, None)
            reader = csv.reader(file)
            pages = list(reader)
        logging.info("PAGES: " + str(pages))
    except IOError:
        logging.warning("No input file was found at " + input_file)
    return pages


def update_pages(csv_file, config, ini_file):  # also updates log_file
    adder_config = config["ADDER"]
    last_time = adder_config["last_request_unix"]
    with open(adder_config["new_pages_file"], "r") as file:
        data = file.readlines()
    print(str(len(data)))
    for i in range(len(data)):
        if TIME_REGEX.search(data[i]):
            if int(data[i].strip().strip("#")) > int(last_time):
                data = data[i + 1::]
                logging.info("NEW PAGES FOUND:\n" + str(data))
                break
    added = []
    new_time = int(last_time)
    j = 0
    for i in range(len(data)):
        if TIME_REGEX.search(data[i]):
            if int(data[i].strip().strip("#")) > new_time:
                new_time = int(data[i].strip().strip("#"))
                if len(data[j + 1:i:]) > 3:
                    added.append(data[j:i:])
                else:
                    warning_msg = (
                        "INCORRECTLY FORMATTED DATA IN "
                        + str(adder_config["new_pages_file"])
                        + " AT LINE "
                        + str(i)
                    )
                    logging.warning(warning_msg)
            j = i
    lines = []
    for request in added:
        channel_name = request[2].strip()
        channel_id = request[3].strip()
        links = request[4::]
        for link in links:
            page_name = get_page_name(link.strip())
            human_name = page_name + " @ " + channel_name
            line = [
                str(human_name),
                str(page_name),
                str(link.strip()),
                "0",
                str(channel_id),
            ]
            lines.append(line)
    with open(csv_file, "a", encoding="utf_8", newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerows(lines)
    adder_config["last_request_unix"] = str(new_time)
    with open(ini_file, "w") as file:
        config.write(file)


def main():
    args = argument_parser()  # command line arguments
    config = config_parser(args["ini_file"])
    basic_config, log_config = (
        config["BASIC"],
        config["LOG"],
    )
    if args["DEBUG_MODE"]:
        log_config["debug_level"] = "DEBUG"
    configure_logging(log_config)
    pages_file, token = basic_config["pages_file"], basic_config["bot_token"]
    logging.info("LOADED FB PAGES FROM FILE: " + pages_file)
    logging.info("LOADED TOKEN: " + token)
    try:
        while True:  # the main loop
            update_pages(pages_file, config, args["ini_file"])
            pages = gather_data(pages_file)
            for page in pages:
                # [0]is HUMAN REDABLE data, [1] is NAME,
                # [2] is URL, [3] is LAST_TIME and [4] is ID
                page_name = page[1]
                human_data = page[0].encode(
                    "ascii", "ignore").decode("ascii", "ignore")
                logging.info("HUMAN DATA: %s", human_data)

                shown_name = page_name.encode(
                    "ascii", "ignore").decode("ascii", "ignore")
                logging.info("SHOWN ON CHANNEL: %s", shown_name)

                url = page[2]
                logging.info("PAGE URL: " + url)
                channel_id = page[4]
                last_time = page[3]
                soup = generate_soup(url)
                # seems to be hardcoded in FB's HTML code to define posts
                posts = soup.find_all("div", "_427x")
                posts.reverse()
                print("HANDLING " + str(page[0]))
                page[3] = new_posts_handling(
                    posts,
                    last_time,
                    token,
                    channel_id,
                    page_name,
                )
                update_csv(pages, pages_file)
            date = get_date()
            logging.info("Now sleeping, time: " + date)
            print("Now sleeping, time: " + date)
            time.sleep(int(basic_config["interval_between_updates"]))
    except Exception as e:
        logging.critical("ERROR AT " + get_date() + "\nERROR INFO:" + str(e))

        raise e


if __name__ == '__main__':
    main()
