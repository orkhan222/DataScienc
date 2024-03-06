# Task5

# I have written a function for each of the 1st methods.
# Import
import re
#* Long

def long(password):
    return len(password) >= 8
#* Upper

def upper(password):
    return re.search(r'[A-Z]', password) is not None
#* Lower

def lower(password):
    return re.search(r'[a-z]', password) is not None
#* Digit

def digit(password):
    return re.search(r'\d', password) is not None
#* Special Characters

def special(password):
    return re.search(r'[!@#$%^&*()]', password) is not None

def check_password(password):
    if (long(password), upper(password), lower(password), digit(password), special(password)):
        return "Strong"
    else:
        return "Not strong"

passwords = ["Password123!", "weakpass", "StrongerPass@123", "12345678", "NoSpecialChar123"]

#* Checking  password
for password in passwords:
    print(f"Password '{password}' is {check_password(password)}.")



print('='*100)
# --------------------------------------------------------------------------------------------------------------------------------------------
#* I have written a function for each of the 2nd methods.


def check_password(password):
    #* long
    if (len(password) >= 8 and
        #* Upper
        re.search(r'[A-Z]', password) and
        #* Lower
        re.search(r'[a-z]', password) and
        # Digit
        re.search(r'\d', password) and
        #* Special Characters
        re.search(r'[!@#$%^&*()]', password)):
        return "Strong"
    else:
        return "Not strong"

passwords = ["Password123!", "weakpass", "StrongerPass@123", "12345678", "NoSpecialChar123"]

for password in passwords:
    print(f"Password '{password}' is {check_password(password)}.")

# --------------------------------------------------------------------------------------------------------------------------------------------
print('='*100)
# I wrote a function for each of the 3rd methods.

def check_password(passwords):
    # Special Characters
    character = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()]).{8,}$')

    results = {}
    for password in passwords:
        if character.match(password):
            results[password] = "Strong"
        else:
            results[password] = "Not strong"
    
    return results

passwords = ["Password123!", "weakpass", "StrongerPass@123", "12345678", "NoSpecialChar123"]
results = check_password(passwords)

for password, strength in results.items():
    print(f"{password}: is {strength}")