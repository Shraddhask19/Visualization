# from urllib.request import urlopen
# from bs4 import BeautifulSoup as soup
# import re

# page = urlopen("https://fast.com/")
# bsobj = soup(page.read())

# for link in bsobj.findAll('a'):
#     if 'href' in link.attrs:
#         print(link.attrs['href'])




# import logging
# from urllib.parse import urljoin
# import requests
# from bs4 import BeautifulSoup

# logging.basicConfig(
#     format='%(asctime)s %(levelname)s:%(message)s',
#     level=logging.INFO)

# class Crawler:

#     def _init_(self, urls=[]):
#         self.visited_urls = []
#         self.urls_to_visit = urls

#     def download_url(self, url):
#         return requests.get(url).text

#     def get_linked_urls(self, url, html):
#         soup = BeautifulSoup(html, 'html.parser')
#         for link in soup.find_all('a'):
#             path = link.get('href')
#             if path and path.startswith('/'):
#                 path = urljoin(url, path)
#             yield path

#     def add_url_to_visit(self, url):
#         if url not in self.visited_urls and url not in self.urls_to_visit:
#             self.urls_to_visit.append(url)

#     def crawl(self, url):
#         html = self.download_url(url)
#         for url in self.get_linked_urls(url, html):
#             self.add_url_to_visit(url)

#     def run(self):
#         while self.urls_to_visit:
#             url = self.urls_to_visit.pop(0)
#             logging.info(f'Crawling: {url}')
#             try:
#                 self.crawl(url)
#             except Exception:
#                 logging.exception(f'Failed to crawl: {url}')
#             finally:
#                 self.visited_urls.append(url)

# if _name_ == '_main_':
#     Crawler(urls=['https://www.imdb.com/']).run()




import urllib, re, mechanize
from urllib.parse import urlparse
from threading import Thread
from bs4 import BeautifulSoup 
from readability.readability import Document

url = "http://walchandsangli.ac.in/"

def scrapper(root, steps):
    urls = [root]
    visited = [root]
    counter = 0

    while counter < steps:
        step_url = scrapeStep(urls)
        urls = []
        for u in step_url:
            if u not in visited:
                urls.append(u)
                visited.append(u)
        counter += 1

    return visited

def scrapeStep(root):
    result_urls = []
    br = mechanize.Browser()
    br.set_handle_robots(False)
    br.addheaders = [('User-agent', 'Firefox')]

    for url in root:
        try:
            br.open(url)
            for link in br.links():
                newurl = urlparse.urljoin(link.base_url, link.url)
                result_urls.append(newurl)
        except:
            print("Error")
    return result_urls


d = {}
threadlist = []

def getReadableArticle(url):
    br = mechanize.Browser()
    br.set_handle_robots(False)
    br.addheaders = [('User-agent', 'Firefox')]

    html = br.open(url).read()

    readable_article = Document(html).summary()
    readable_title = Document(html).short_title()

    soup = BeautifulSoup(readable_article)

    final_article = soup.text

    links = soup.findAll('img', src=True)

    title_article = []
    title_article.append(final_article)
    title_article.append(readable_title)
    return title_article


def dungalo(urls):
    article_text = getReadableArticle(urls)[0]
    d[urls] = article_text


def getMultiHtml(urlsList):

    for urls1 in urlsList:
        try:
            t = Thread(target=dungalo, args=(urls1, ))
            threadlist.append(t)
            t.start()
        except:
            nnn = True

    for g in threadlist:
        g.join()

    return d

print(scrapper(url,8))
