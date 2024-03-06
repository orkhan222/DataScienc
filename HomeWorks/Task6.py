# Task 6
import re

def simple(text):
    date_pattern = r'\b\d{2}[/-]\d{2}[/-]\d{4}\b|\b\d{4}[/-]\d{2}[/-]\d{2}\b'
    return re.findall(date_pattern, text)


text_to_search = "Important dates: 23/04/2021, 01-05-2020, and 2020-12-31. Also, 7-7-2007 is noted."


found = simple(text_to_search)
print("Dates found:", found)

