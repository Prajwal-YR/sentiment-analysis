# This program does:
# Scrape the Instagram images from a list of users
# Apply sentiment analysis on these images and store the corresponded labels in a dictionary

# pip install instagram-scraper first
import os
import io
from google.cloud import vision
from google.cloud.vision import types
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append("/anaconda3/lib/python3.7/site-packages")
sys.path.append("/usr/local/lib/python3.7/site-packages")


# change the following path to the google vision api credential path
os.environ['GOOGLE_APPLICATION_CREDENTIALS']='C:\Learn\learn4\ocrObject-6f825255c242.json'
###

client = vision.ImageAnnotatorClient()

# have to pip install google-cloud-vision

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


# test first
def get_face(image):
    response=client.face_detection(image=image)
    faces = response.face_annotations
    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    flag=0
    label=1
    cause=None
    print('\nFaces:')
    for face in faces:
        print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
        print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
        print('sorrow: {}'.format(likelihood_name[face.sorrow_likelihood]))
        if face.anger_likelihood in [3,4,5]:
            label = 0#negate
            flag=1
            cause ="Anger"
        if face.sorrow_likelihood in [3,4,5]:
            label = 0#negate
            flag=1
            cause = "Sorrow"
        if face.joy_likelihood in [3,4,5]:
            label = 2#positive
            flag=1
            cause = "Happy"
        if flag==1:
            break
    return [label,cause]


def get_text(image):
    response = client.text_detection(image=image)
    texts = response.text_annotations
    temp=[]
    print('\nText:')
    for text in texts:
        temp.append(text.description)
    return temp



def vision_face(file_name):
    im=Image.open(file_name)
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()
    plt.imshow(im)
    image = types.Image(content=content)
    #print(content)
    #for text in get_text(image):
    try:
        print(get_text(image)[0]) #get
    except:
        print("")
    label,cause=get_face(image)  #get
    if label==0:
        print("\nRESULT : Negative")
    elif label==1:
        print("\nRESULT : Neutral")
    else:
        print("\nRESULT : Positive")
    print("CAUSE:",cause)
    return label


image_dict = {}

for path in path_list:
	count = 0
	username = os.path.basename(path)
	for image in os.listdir(path):
		direct = os.getcwd()
		image = os.path.join(path,image)
		label = vision_face(image)
		image_dict[username+string(count)] = label
		count = count + 1



