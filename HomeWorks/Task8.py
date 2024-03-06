# Task 8

import re

post = "Join the conversation on #AI and #MachineLearning. Follow us @DataWorld!"

hashtag = r'#\w+'
mention = r'@\w+'


hashtags = re.findall(hashtag, post)
mentions = re.findall(mention, post)

print("Hashtags:", hashtags)
print("Mentions:", mentions)