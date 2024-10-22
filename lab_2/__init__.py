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
from tutorial.items import YyItem
from scrapy import log

class YySpider(scrapy.Spider):
    name = 'yy'
    allowed_domains = ['*.z.yy.com']
    start_urls = [
        "http://video.z.yy.com/getVideoTapeByPid.do?uid=314827531&programId=15012x01_314827531_1424699060&videoFrom=popularAnchor",
        ]
    
    base_url = 'http://video.z.yy.com/getVideoTapeByPid.do'
    def parse(self, response):
        time.sleep(1)
        playList = response.xpath('//ul[@id="playList"]/li')
        items = []
        urls = []           # 
        zone_urls = []      # 访问fan's zone url
        author_urls = []     # 访问author's zone url
        for video in playList:
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
            try:
                # 确定是否要根据 data_pid 进行popularAuthor 扩展
                with open( item['data_anchor'] + '.txt', 'r') as fp:
                    if item['data_pid'] in fp.read().decode('utf8'):
                        pass
                    else:
                        urls.append("http://video.z.yy.com/getVideoTapeByPid.do?" +
                                    "uid=" + item["data_anchor"] +
                                    "&programId=" + item["data_pid"] +
                                    "&videoFrom=popularAnchor")           
            except IOError as e:
                if e.errno == 2:
                    open( item['data_anchor'] + '.txt', 'a').close()
            
            # 确定是否解析到主播的 data_puid, 如果解析到了，就将其保存到author.txt      
            try:
                if item['data_puid']:
                    with open( 'author.txt', 'a+') as fp:
                        if item['data_puid'] in fp.read():
                            pass
                        else:
                            fp.seek(0, 2)
                            fp.write('|'.join((item['data_anchor'], item['data_puid'])) + '\n' )
            except:
                pass
            # 确定是否要抓取粉丝的zone, 如果需要构造请求url
            try:
                with open('crawled_ouid.txt', 'r') as fp:
                    if item['data_ouid'] in fp.read():
                        pass
                    else:
                        zone_urls.append("http://z.yy.com/zone/myzone.do?type=2" + 
                                         "&puid=" + item['data_ouid'] +
                                         "&feedFrom=1&chkact=0" )
            except IOError as e:
                if e.errno == 2:
                    open('crawled_ouid.txt', 'a').close()
            # 确定是否要访问主播的zone， 如果需要则构造请求url
            try:
                with open('all_author.txt', 'r') as fp:
                    if item['data_anchor'] in fp.read().decode('utf8'):
                        pass
                    else:
                        author_url = "http://z.yy.com/zone/myzone.do?type=10" + "&puid=" + item['data_puid']
                        author_urls.append(author_url)
            except IOError as e:
                if e.errno ==2:
                    open('all_author.txt', 'a').close()
        #items.extend([self.make_requests_from_url(url).replace(callback=self.parse) for url in urls])
        items.extend([self.make_requests_from_url(url).replace(callback=self.parse_ex) for url in zone_urls])
        items.extend([self.make_requests_from_url(url).replace(callback=self.parse_author) for url in author_urls])
        
        return items
    
    def parse_author(self, response):
        """ 解析主播的信息(只有主播的地址才需要这条解析） """
        try:
            items = []
            loadPage = response.xpath('//script')[-1]
            user = loadPage.re(r'user:(\{\s*.*\})')[0]
            user = re.sub(r'\n', '', user)              # 替换掉空白符，后面的一条记录方便以换行符标记
            user = json.loads(user)
            user_id = user['uid']
            user_yyNo = user['yyNo']
            user_uid = user['UID']
            user_json = json.dumps(user)
            if user_id and user_uid and user_yyNo:
                with open('all_author.txt', 'a+') as fp:
                    if user_id not in fp.read().decode('utf8'):
                        fp.seek(0, 2)
                        fp.write('|'.join((user_id, user_yyNo, user_uid, user_json)) + '\n')
            else:
                with open('unsolved_anchor.txt', 'a+') as fp:
                    fp.seek(0, 2)
                    fp.write('|'.join((user_id, user_yyNo, user_uid, user_json)) + '\n')
        except :
            pass
        
        return items

    def parse_feed(self, response):
        "根据feedid, 粉丝的puid找到视频的data_pid, 主播的data_puid"
        try:
            items = []
            v_meta = response.xpath('//div[@class="v-meta"]')
            data_puid = v_meta.xpath('./span[@class="v-performer"]/a').re(r'=(\w+)')[0]
            data_pid = v_meta.xpath('.//li[@id="weibo"]/@data-pid').extract()[0]
            (data_ouid, v_feedId) = response.url.split('=')[-2:]
            item = YyItem()
            item['data_puid'] = data_puid
            item['data_pid'] = data_pid
            item['data_ouid'] = data_ouid
            item['v_feedId'] = v_feedId
            log.msg('parse_feed caller ==='+ item['data_pid'], level=log.WARNING)
            url = "http://video.z.yy.com/getVideoTapeByPid.do?" + "programId=" + item["data_pid"] + "&videoFrom=popularAnchor"
            items.extend([self.make_requests_from_url(url).replace(callback=self.parse) ])     
        except IndexError as e:
            with open('feedID_not_performer.txt', 'a+') as fp:
                fp.seek(0, 2)
                fp.write('|'.join(response.url.split('=')[-2]) + '\n')
                #fp.write(item['data_ouid'] + '|' + item['v_feedId'] + '\n')
        
        return items
    
    def parse_ex(self, response):
        """ 从页面的最后一个<script>中找数据 """
        time.sleep(1)
        loadPage = response.xpath('//script')[-1]
        # 后面这个try是为了捕获未开YY空间自动跳转产生的错误
        try:
            user = loadPage.re(r'user:(\{\s*.*\})')[0]
            user = json.loads(user)
            user_id = user['uid']
            user_uid = user['UID']
            yyNo = user['yyNo']
            playList = loadPage.re(r'videoData:(\[\s*.*\])')[0]
            playList = json.loads(playList)
            items = []
            urls = []
        except IndexError:
            if "type=10" in response.url:
                return []            
        feedId_urls = []
        for video in playList:
            item = YyItem()
            item['data_anchor'] = ''
            if video['performerNames']:
                item['data_anchor'] = video['performerNames'].keys()[0]
            item['data_ouid'] = str(user_uid)
            item['data_oid'] = str(user_id)
            item['data_pid'] = video['programId']
            item['data_yyNo'] = str(yyNo)
            item['v_thumb_img'] = video['snapshotUrl']
            item['v_play_times'] = str(video['playTimes'])
            item['v_duration'] = str(video['duration'])
            item['v_feedId'] = str(video['feedId'])
            items.append(item)
            
            if not item['data_pid']:
                try:
                    with open(  item['data_anchor'] + '.txt', 'r') as fp:
                        if item['data_pid'] in fp.read().decode('utf8'):
                            continue
                        else:
                            urls.append("http://video.z.yy.com/getVideoTapeByPid.do?" + 
                                        "uid=" + item['data_anchor'] +
                                        "@programId=" + item['data_pid'] +
                                        "&videoFrom=popularAnchor#" )
                except IOError as e:
                    open(  item['data_anchor'] + '.txt', 'a+').close()
            elif item['v_feedId']:
                feedId_urls.append("http://z.yy.com/zone/myzone.do?" + 
                                   "puid=" + item['data_ouid'] +
                                   "&feedId=" + item['v_feedId'])
            else:
                log.msg(response.url + '===programId, feedId', level="WARNING")
            try:
                with open( 'crawled_ouid.txt', 'a+') as fp:
                    if item['data_ouid'] not in fp.read():
                        # 从文件末尾处追加
                        fp.seek(0, 2)
                        fp.write(item['data_ouid'] + '\n')
                        log.msg('===' + item['data_ouid'] + 'had been crawled!', level=log.WARNING)
            except IOError as e:
                if e.errno == 2:
                    open( 'crawled_ouid.txt', 'a').close()
                else:
                    log.msg('===' + item['data_ouid'] + 'some where song', level=log.WARNING)
                    
        items.extend([self.make_requests_from_url(url).replace(callback=self.parse) for url in urls])
        items.extend([self.make_requests_from_url(url).replace(callback=self.parse_feed) for url in feedId_urls])
        
        return items

        

class ExtendYySpider(scrapy.Spider):
    time.sleep(random.randint(1, 5))
    name = 'extendYy'
    allowed_domains = ['z.yy.com']
    start_urls = [
        "http://z.yy.com/zone/myzone.do?type=2&puid=15d77dc1e4f8d52b90cc4e478ec81db9&feedFrom=1&chkact=1",
        ]
    
    def parse(self, response):
        """ 从页面的最后一个<script>中找数据 """
        time.sleep(1)
        loadPage = response.xpath('//script')[-1]
        user = loadPage.re(r'user:(\{\s*.*\})')[0]
        user = json.loads(user)
        uid = user['uid']
        yyNo = user['yyNo']
        playList = loadPage.re(r'videoData:(\[\s*.*\])')[0]
        playList = json.loads(playList)
        items = []
        urls = []
        for video in playList:
            item = YyItem()
            item['data_anchor'] = ''
            if video['performerNames']:
                item['data_anchor'] = video['performerNames'].keys()[0]
                log.msg(str(uid)+ 'res not have performerNames', level=log.WARNING)
            item['data_ouid'] = str(uid)
            item['data_oid'] = str(video['ownerId'])
            item['data_pid'] = video['programId']
            item['data_yyNo'] = str(yyNo)
            item['v_thumb_img'] = video['snapshotUrl']
            item['v_play_times'] = str(video['playTimes'])
            item['v_duration'] = str(video['duration'])
            items.append(item)
            try:
                with open( item['data_anchor'] + '.txt', 'r') as fp:
                    if item['data_pid'] in fp.read().decode('utf8'):
                        continue
                    else:
                        urls.append("http://video.z.yy.com/getVideoTapeByPid.do?" + 
                                    "uid=" + item['data_anchor'] +
                                    "@programId=" + item['data_pid'] +
                                    "&videoFrom=popularAnchor#" )
            except IOError as e:
                open( item['data_anchor'] + '.txt', 'a+').close()
                
            try:
                with open('crawled_ouid.txt', 'a+') as fp:
                    if item['data_ouid'] in fp.read():
                        pass
                    else:
                        fp.write(item['data_ouid'] + '\n')
            except IOError as e:
                if e.errno == 2:
                    open('crawled_ouid.txt', 'a').close()
                    

        items.extend([self.make_requests_from_url(url).replace(callback=YySpider().parse) for url in urls])
        
        return items
                    
            
            
           