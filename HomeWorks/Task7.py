import re

text_to_search = "My contacts are 123-456-7890, 987-654-3210. Call soon."

pattern = r'\d{3}-\d{3}-\d{4}'

phone_numbers = re.findall(pattern, text_to_search)

print(phone_numbers)
