# -*- coding: utf-8 -*-
# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.

import json
import os
import re
import time
import random
import scrapy
import logging
from tutorial.items import YyItem
from scrapy import log

logging.basicConfig(
    filename='spider.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class YySpider(scrapy.Spider):
    name = 'yy'
    allowed_domains = ['*.z.yy.com']
    start_urls = [
        "http://video.z.yy.com/getVideoTapeByPid.do?uid=314827531&programId=15012x01_314827531_1424699060&videoFrom=popularAnchor",
        ]
    
    base_url = 'http://video.z.yy.com/getVideoTapeByPid.do'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.info(f"Spider '{self.name}' initialized with start URLs: {self.start_urls}")
        
    def parse(self, response):
        logging.info(f"Received response for URL: {response.url}")

        playList = response.xpath('//ul[@id="playList"]/li')
        if not playList:
            logging.warning(f"No videos found on page: {response.url}")

        items = []
        urls = []
        zone_urls = []  
        author_urls = []  

        logging.info("Starting to parse videos...")
        for video in playList:
            try:
                item = YyItem()
                item['data_anchor'] = video.xpath('@data-anchor').extract()[0]
                item['data_puid'] = video.xpath('@data-puid').extract()[0].strip()
                item['data_videofrom'] = video.xpath('@data-videofrom').extract()[0]
                item['data_pid'] = video.xpath('@data-pid').extract()[0]
                item['data_ouid'] = video.xpath('@data-ouid').extract()[0]
                item['v_thumb_img'] = video.xpath('.//img/@src').extract()[0]
                item['v_word_cut'] = video.xpath('.//a[@class="word-cut"]/text()').extract()[0]
                item['v_play_times'] = video.xpath('.//span[@class="times"]/text()').extract()[0]
                item['v_duration'] = video.xpath('.//span[@class="duration"]/text()').extract()[0]
                item['data_yyNo'] = ''
                items.append(item)

                logging.info(f"Parsed video: {item['data_anchor']} | PUID: {item['data_puid']} | PID: {item['data_pid']}")

                try:
                    with open(item['data_anchor'] + '.txt', 'r') as fp:
                        if item['data_pid'] in fp.read().decode('utf8'):
                            logging.info(f"Video {item['data_pid']} already processed for anchor {item['data_anchor']}")
                        else:
                            url = ("http://video.z.yy.com/getVideoTapeByPid.do?"
                                f"uid={item['data_anchor']}&programId={item['data_pid']}&videoFrom=popularAnchor")
                            urls.append(url)
                            logging.info(f"Added popularAuthor URL: {url}")
                except IOError as e:
                    if e.errno == 2:
                        logging.warning(f"File not found for anchor {item['data_anchor']}.txt. Creating a new file.")
                        open(item['data_anchor'] + '.txt', 'a').close()

                try:
                    if item['data_puid']:
                        with open('author.txt', 'a+') as fp:
                            if item['data_puid'] in fp.read():
                                logging.info(f"PUID {item['data_puid']} already exists in author.txt")
                            else:
                                fp.seek(0, 2)
                                fp.write('|'.join((item['data_anchor'], item['data_puid'])) + '\n')
                                logging.info(f"Saved PUID {item['data_puid']} for anchor {item['data_anchor']} in author.txt")
                except Exception as e:
                    logging.error(f"Error saving PUID {item['data_puid']}: {e}")

                try:
                    with open('crawled_ouid.txt', 'r') as fp:
                        if item['data_ouid'] in fp.read():
                            logging.info(f"OUID {item['data_ouid']} already exists in crawled_ouid.txt")
                        else:
                            zone_url = ("http://z.yy.com/zone/myzone.do?type=2"
                                        f"&puid={item['data_ouid']}&feedFrom=1&chkact=0")
                            zone_urls.append(zone_url)
                            logging.info(f"Added fan zone URL: {zone_url}")
                except IOError as e:
                    if e.errno == 2:
                        logging.warning(f"File crawled_ouid.txt not found. Creating a new file.")
                        open('crawled_ouid.txt', 'a').close()

                try:
                    with open('all_author.txt', 'r') as fp:
                        if item['data_anchor'] in fp.read().decode('utf8'):
                            logging.info(f"Anchor {item['data_anchor']} already exists in all_author.txt")
                        else:
                            author_url = ("http://z.yy.com/zone/myzone.do?type=10"
                                        f"&puid={item['data_puid']}")
                            author_urls.append(author_url)
                            logging.info(f"Added author zone URL: {author_url}")
                except IOError as e:
                    if e.errno == 2:
                        logging.warning(f"File all_author.txt not found. Creating a new file.")
                        open('all_author.txt', 'a').close()

            except Exception as e:
                logging.error(f"Error parsing video in playlist: {e}")

        logging.info(f"Generated {len(urls)} popularAuthor URLs.")
        logging.info(f"Generated {len(zone_urls)} fan zone URLs.")
        logging.info(f"Generated {len(author_urls)} author zone URLs.")

        items.extend([self.make_requests_from_url(url).replace(callback=self.parse_ex) for url in zone_urls])
        items.extend([self.make_requests_from_url(url).replace(callback=self.parse_author) for url in author_urls])

        logging.info(f"Returning {len(items)} items for further processing.")
        return items

    
    def parse_author(self, response):
        logging.info(f"Parsing author data from URL: {response.url}")
        try:
            loadPage = response.xpath('//script')[-1]
            user = loadPage.re(r'user:(\{\s*.*\})')[0]
            user = json.loads(user)
            user_id = user['uid']
            user_yyNo = user['yyNo']
            user_uid = user['UID']
            logging.info(f"Extracted author data: UID={user_uid}, YYNo={user_yyNo}, ID={user_id}")
        except Exception as e:
            logging.error(f"Error parsing author data from {response.url}: {e}")
            return []


    def parse_feed(self, response):
        """
        根据feedid, 粉丝的puid找到视频的data_pid, 主播的data_puid
        """
        logging.info(f"Received response for URL: {response.url}")
        try:
            items = []
            v_meta = response.xpath('//div[@class="v-meta"]')
            logging.debug("Parsing v-meta data...")

            data_puid = v_meta.xpath('./span[@class="v-performer"]/a').re(r'=(\w+)')[0]
            data_pid = v_meta.xpath('.//li[@id="weibo"]/@data-pid').extract()[0]
            data_ouid, v_feedId = response.url.split('=')[-2:]

            logging.info(f"Extracted data: PUID={data_puid}, PID={data_pid}, OUID={data_ouid}, FeedID={v_feedId}")

            item = YyItem()
            item['data_puid'] = data_puid
            item['data_pid'] = data_pid
            item['data_ouid'] = data_ouid
            item['v_feedId'] = v_feedId

            logging.info(f"Created item: {item}")

            url = ("http://video.z.yy.com/getVideoTapeByPid.do?"
                f"programId={item['data_pid']}&videoFrom=popularAnchor")
            logging.info(f"Generated URL for further processing: {url}")

            items.extend([self.make_requests_from_url(url).replace(callback=self.parse)])
            logging.info(f"Added request for URL: {url}")

        except IndexError as e:
            logging.warning(f"IndexError encountered: {e}")
            logging.warning(f"Missing performer data for URL: {response.url}")

            try:
                with open('feedID_not_performer.txt', 'a+') as fp:
                    fp.seek(0, 2)
                    fp.write('|'.join(response.url.split('=')[-2]) + '\n')
                logging.info(f"Logged missing performer data to feedID_not_performer.txt for URL: {response.url}")
            except Exception as file_error:
                logging.error(f"Failed to write to feedID_not_performer.txt: {file_error}")

        logging.info(f"Returning {len(items)} items for further processing.")
        return items

    
def parse_ex(self, response):
    """从页面的最后一个<script>中找数据"""
    logging.info(f"Processing response for URL: {response.url}")
    time.sleep(1)

    try:
        loadPage = response.xpath('//script')[-1]
        logging.debug("Extracted last <script> element from the page.")

        user = loadPage.re(r'user:(\{\s*.*\})')[0]
        user = json.loads(user)
        user_id = user['uid']
        user_uid = user['UID']
        yyNo = user['yyNo']
        logging.info(f"Extracted user data: UID={user_uid}, ID={user_id}, yyNo={yyNo}")

        playList = loadPage.re(r'videoData:(\[\s*.*\])')[0]
        playList = json.loads(playList)
        logging.info(f"Extracted video playlist with {len(playList)} items.")
        items = []
        urls = []

    except IndexError:
        logging.warning(f"No user or video data found in response for URL: {response.url}")
        if "type=10" in response.url:
            logging.info(f"Skipping URL due to missing type=10 condition: {response.url}")
            return []

    feedId_urls = []
    for video in playList:
        item = YyItem()
        item['data_anchor'] = ''
        if video.get('performerNames'):
            item['data_anchor'] = list(video['performerNames'].keys())[0]

        item['data_ouid'] = str(user_uid)
        item['data_oid'] = str(user_id)
        item['data_pid'] = video['programId']
        item['data_yyNo'] = str(yyNo)
        item['v_thumb_img'] = video['snapshotUrl']
        item['v_play_times'] = str(video['playTimes'])
        item['v_duration'] = str(video['duration'])
        item['v_feedId'] = str(video['feedId'])

        logging.debug(f"Created YyItem: {item}")
        items.append(item)

        if not item['data_pid']:
            try:
                with open(item['data_anchor'] + '.txt', 'r') as fp:
                    if item['data_pid'] in fp.read():
                        logging.info(f"Program ID {item['data_pid']} already processed for anchor {item['data_anchor']}.")
                        continue
                    else:
                        url = (f"http://video.z.yy.com/getVideoTapeByPid.do?"
                               f"uid={item['data_anchor']}@programId={item['data_pid']}&videoFrom=popularAnchor#")
                        urls.append(url)
                        logging.info(f"Generated URL for missing program ID: {url}")
            except IOError as e:
                if e.errno == 2:
                    open(item['data_anchor'] + '.txt', 'a+').close()
                    logging.warning(f"File {item['data_anchor']}.txt not found, created a new one.")
        elif item['v_feedId']:
            feedId_url = (f"http://z.yy.com/zone/myzone.do?"
                          f"puid={item['data_ouid']}&feedId={item['v_feedId']}")
            feedId_urls.append(feedId_url)
            logging.info(f"Generated Feed ID URL: {feedId_url}")
        else:
            logging.warning(f"Missing program ID and Feed ID for video: {video}")

        try:
            with open('crawled_ouid.txt', 'a+') as fp:
                if item['data_ouid'] not in fp.read():
                    fp.seek(0, 2)
                    fp.write(item['data_ouid'] + '\n')
                    logging.info(f"Marked OUID {item['data_ouid']} as crawled.")
        except IOError as e:
            if e.errno == 2:
                open('crawled_ouid.txt', 'a').close()
                logging.warning(f"File crawled_ouid.txt not found, created a new one.")
            else:
                logging.error(f"Failed to update crawled OUID file: {e}")

    items.extend([self.make_requests_from_url(url).replace(callback=self.parse) for url in urls])
    logging.info(f"Added {len(urls)} requests for program URLs.")

    items.extend([self.make_requests_from_url(url).replace(callback=self.parse_feed) for url in feedId_urls])
    logging.info(f"Added {len(feedId_urls)} requests for Feed ID URLs.")

    logging.info(f"Returning {len(items)} items for further processing.")
    return items

        

class ExtendYySpider(scrapy.Spider):
    time.sleep(random.randint(1, 5))
    name = 'extendYy'
    allowed_domains = ['z.yy.com']
    start_urls = [
        "http://z.yy.com/zone/myzone.do?type=2&puid=15d77dc1e4f8d52b90cc4e478ec81db9&feedFrom=1&chkact=1",
    ]

    def parse(self, response):
        """从页面的最后一个<script>中找数据"""
        logging.info(f"Processing response for URL: {response.url}")
        time.sleep(1)
        try:
            loadPage = response.xpath('//script')[-1]
            logging.debug("Extracted last <script> element from the page.")

            user = loadPage.re(r'user:(\{\s*.*\})')[0]
            user = json.loads(user)
            uid = user['uid']
            yyNo = user['yyNo']
            logging.info(f"Extracted user data: UID={uid}, yyNo={yyNo}")

            playList = loadPage.re(r'videoData:(\[\s*.*\])')[0]
            playList = json.loads(playList)
            logging.info(f"Extracted video playlist with {len(playList)} items.")
            items = []
            urls = []

            for video in playList:
                item = YyItem()
                item['data_anchor'] = ''
                if video.get('performerNames'):
                    item['data_anchor'] = list(video['performerNames'].keys())[0]
                else:
                    logging.warning(f"Video has no performerNames: UID={uid}")

                item['data_ouid'] = str(uid)
                item['data_oid'] = str(video['ownerId'])
                item['data_pid'] = video['programId']
                item['data_yyNo'] = str(yyNo)
                item['v_thumb_img'] = video['snapshotUrl']
                item['v_play_times'] = str(video['playTimes'])
                item['v_duration'] = str(video['duration'])
                items.append(item)

                try:
                    with open(item['data_anchor'] + '.txt', 'r') as fp:
                        if item['data_pid'] in fp.read():
                            logging.info(f"Program ID {item['data_pid']} already processed for anchor {item['data_anchor']}.")
                            continue
                        else:
                            url = (f"http://video.z.yy.com/getVideoTapeByPid.do?"
                                   f"uid={item['data_anchor']}@programId={item['data_pid']}&videoFrom=popularAnchor#")
                            urls.append(url)
                            logging.info(f"Generated URL for missing program ID: {url}")
                except IOError as e:
                    if e.errno == 2:
                        open(item['data_anchor'] + '.txt', 'a+').close()
                        logging.warning(f"File {item['data_anchor']}.txt not found, created a new one.")

                try:
                    with open('crawled_ouid.txt', 'a+') as fp:
                        if item['data_ouid'] not in fp.read():
                            fp.write(item['data_ouid'] + '\n')
                            logging.info(f"Marked OUID {item['data_ouid']} as crawled.")
                except IOError as e:
                    if e.errno == 2:
                        open('crawled_ouid.txt', 'a').close()
                        logging.warning(f"File crawled_ouid.txt not found, created a new one.")

            items.extend([self.make_requests_from_url(url).replace(callback=YySpider().parse) for url in urls])
            logging.info(f"Added {len(urls)} requests for program URLs.")

        except IndexError as e:
            logging.error(f"IndexError while processing response for URL {response.url}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error in parse method for URL {response.url}: {e}")

        logging.info(f"Returning {len(items)} items for further processing.")
        return items
                    
def close(self, reason):
    logging.info(f"Spider '{self.name}' closed. Reason: {reason}")
    super().close(reason)            
            
           