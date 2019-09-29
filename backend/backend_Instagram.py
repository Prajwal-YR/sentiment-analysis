
# pip install instagram-scraper
import os
Instagram_username = ['athinkingneal','_peebz_']

path_list = []

def scrapeInstagram(username):
	os.system("instagram-scraper " + username + " -m 40 --proxies true")
	# scrape maximum 40 pictures
	current_path = os.getcwd()
	image_folder = os.path.join(current_path,username)
	path_list.append(image_folder)

for username in Instagram_username:
	scrapeInstagram(username)
	# the images are stored into <current working directory>/<username>

#print(path_list)
