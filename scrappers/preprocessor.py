
#p = "#loving I have a bad day :( @alice https://1123"
#p = "#Colleges See Rise In #Mental Health Issues  http://su.pr/AcXyhz RT @nprnews #student #health #therapy #anxiety #depression #stigma #stress"
p = "RT @Eman_Gway    Jus got sexually harrased by a student. HELLO! GOOD MORNING! Welcome to Alternative School!! #AGONY<<<HAHAlarious!"

import re
def clean_tweet(tweet):
    tweet = re.sub('http\S+\s*', '', tweet)  # remove URLs
    tweet = re.sub('RT|cc', '', tweet)  # remove RT and cc
    tweet = re.sub('#\S+', '', tweet)  # remove hashtags
    tweet = re.sub('@\S+', '', tweet)  # remove mentions
    tweet = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', tweet)  # remove punctuations
    tweet = re.sub('\s+', ' ', tweet)  # remove extra whitespace
    return tweet

p = clean_tweet(p)
print(p)
