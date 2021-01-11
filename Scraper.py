from selenium import webdriver
import csv
import time
from Scroller import scroll


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
posts.pop(0)
for t in posts:
    t = t.text

category.pop(0)
for t in category:
    t = t.text

date_posted.pop(0)
for t in date_posted:
    t = t.text

last_reply.pop(0)
for t in last_reply:
    t = t.text

replies.pop(0)
for t in replies:
    t = t.text

user.pop(0)
for t in user:
    t = t.text

driver.close()
