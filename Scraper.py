from selenium import webdriver
import csv
import time
import re

from Scroller import scroll
from Analysis import load_data

URL = 'https://discussion.dreamhost.com/latest'
PATH = 'M:\Programming\Python Support Files\chromedriver_win32\chromedriver.exe'
driver = webdriver.Chrome(PATH)

scroll(URL, driver)

#scrapes categories
posts = driver.find_elements_by_class_name('title')
category = driver.find_elements_by_class_name('category-name')
dates = driver.find_elements_by_class_name('relative-date') #gets date posted and last post
replies = driver.find_elements_by_class_name('number')
user = driver.find_elements_by_class_name('editor')


#splits dates
date_posted = []
last_reply = []
index = 0
for d in dates:
    if index % 2 == 0:
        date_posted.append(d)
    else:
        last_reply.append(d)
    index += 1

#inefficent method but it just works  ¯\_(ツ)_/¯

#delete first and turn to text
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\'\“\”\’\|@,;]')

def clean(t):
    text = t.text
    text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    return text

posts.pop(0)
cleaned_posts = [clean(t) for t in posts]

category.pop(0)
cleaned_categories = [clean(t) for t in posts]

date_posted.pop(0)
cleaned_dates = []
for t in date_posted:
    cleaned_dates.append(t.text)

last_reply.pop(0)
cleaned_last_reply = []
for t in last_reply:
    cleaned_last_reply.append(t.text)

replies.pop(0)
cleaned_replies = []
for t in replies:
    cleaned_replies.append(t.text)

user.pop(0)
cleaned_users = []
for t in user:
    cleaned_users.append(t.text)

with open('data.csv', mode= 'w', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)

    writer.writerows(zip(cleaned_posts, cleaned_categories, cleaned_dates, cleaned_last_reply, cleaned_replies, cleaned_users))

load_data()

driver.close()
