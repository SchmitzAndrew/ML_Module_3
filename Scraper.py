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
cleaned_categories = [clean(t) for t in category]

date_posted.pop(0)
cleaned_dates = [clean(t) for t in date_posted]

last_reply.pop(0)
cleaned_last_reply = [clean(t) for t in last_reply]

replies.pop(0)
cleaned_replies = [clean(t) for t in replies]

user.pop(0)
cleaned_users = [clean(t) for t in user]

with open('data.csv', mode= 'w', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(zip(cleaned_posts, cleaned_categories, cleaned_dates, cleaned_last_reply, cleaned_replies, cleaned_users))

load_data()

driver.close()
