from selenium import webdriver
import csv
import time
from Scroller import scroll

URL = 'https://discussion.dreamhost.com/latest'
PATH = 'M:\Programming\Python Support Files\chromedriver_win32\chromedriver.exe'
driver = webdriver.Chrome(PATH)

scroll(URL)

driver.close
