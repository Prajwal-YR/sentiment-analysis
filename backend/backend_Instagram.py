# pip install instagram-scraper
import os
Instagram_username = ['athinkingneal']
def scrapeInstagram(username):
	os.system("instagram-scraper username -m 40")
	# scrape maximum 40 pictures

for username in Instagram_username:
	scrapeInstagram(username)
	# the images are stored into <current working directory>/<username>
