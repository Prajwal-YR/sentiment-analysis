import pandas as pd
twitter_username = ['noneprivacy','iamsrk']
def scrapeTwitter(username,limit):
    import sys
    sys.path.append("/usr/local/lib/python3.7/site-packages")
    import twint
    # notice the package path problem, I use the above commands to add the twint package into my current path
    
    c = twint.Config()
    c.Username = username
    c.Output = username + ".csv"
    c.Store_csv = True
    c.Limit = limit
    twint.run.Search(c)



for user_name in twitter_username:
    #print(type(user_name))
    scrapeTwitter(user_name, 10)
    userDf = pd.read_csv(user_name + ".csv") # convert the csv file into a panda dataframe
    student_twitter = userDf['tweet'].tolist() # list of tweets of current user in the loop
    #print(student_twitter)
