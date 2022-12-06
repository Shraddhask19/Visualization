import streamlit as st
import pandas as pd
import numpy as np
from urllib.request import urljoin
from bs4 import BeautifulSoup
import requests
from urllib.request import urlparse

# global uploaded_file

def app():
    st.title("Web Mining")
    st.subheader("Crawler")
    links_intern = set()

    c1, c2 = st.columns(2)

    with c1:
        text_input = st.text_input(
            "Input URL for crawling",
            "http://www.fast.com",
            key="placeholder",
        )

    with c2:
        depth = st.text_input('Depth: ', '1')

    if st.button('Crawl'):
        st.write("Hyperlinks Are: ")
        input_url = text_input
        depth = int(depth)

        links_extern = set()

        def level_crawler(input_url):
            temp_urls = set()
            current_url_domain = urlparse(input_url).netloc

            beautiful_soup_object = BeautifulSoup(
                requests.get(input_url).content, "lxml")

            for anchor in beautiful_soup_object.findAll("a"):
                href = anchor.attrs.get("href")
                if(href != "" or href != None):
                    href = urljoin(input_url, href)
                    href_parsed = urlparse(href)
                    href = href_parsed.scheme
                    href += "://"
                    href += href_parsed.netloc
                    href += href_parsed.path
                    final_parsed_href = urlparse(href)
                    is_valid = bool(final_parsed_href.scheme) and bool(
                        final_parsed_href.netloc)
                    if is_valid:
                        if current_url_domain not in href and href not in links_extern:
                            st.write("Extern - {}".format(href))
                            links_extern.add(href)
                        if current_url_domain in href and href not in links_intern:
                            st.write("Intern - {}".format(href))
                            links_intern.add(href)
                            temp_urls.add(href)
            return temp_urls


        if(depth == 0):
            st.write("Intern - {}".format(input_url))

        elif(depth == 1):
            level_crawler(input_url)

        else:
            queue = []
            queue.append(input_url)
            for j in range(depth):
                for count in range(len(queue)):
                    url = queue.pop(0)
                    urls = level_crawler(url)
                    for i in urls:
                        queue.append(i)


    st.write('---')

    st.subheader('PageRank')

    