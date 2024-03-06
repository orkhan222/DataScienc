
# Task3
data = [
    {"name": "Orkhan", "age": 25, "salary": 700},
    {"name": "Ali", "age": 30, "salary": 1800},
    {"name": "Veli", "age": 35, "salary": 500},
    {"name": "Fidan", "age": 28, "salary": 1200},
    {"name": "Zeynab", "age": 40, "salary": 3000},
    {"name": "Flora", "age": 32, "salary": 5000},
    {"name": "Kerim", "age": 19, "salary": 300},
    {"name": "Flora", "age": 32, "salary": 5000},
    {"name": "Adil", "age": 49, "salary": 1660},
    {"name": "Asmar", "age": 22, "salary": 1200}
]
# Filter
filtered = list(filter(lambda x: x["age"] >= 30, data))

# Mean Salary
mean = sum(map(lambda x: x["salary"], data)) / len(data)

# Lowercase1
def to_lowercase(data):
    lowercase = map(lambda x: x["name"].lower(), data)
    return lowercase
fiters = to_lowercase(data)
# Lowercase 2
lowercase = list(map(lambda x: {**x, "name": x["name"].lower()}, data))

# Prints 
print('='*80)

print("Filtered  (Age >= 30):", filtered)
print('='*80)

print("Mean Salary:", mean)
print('='*80)

print(" Lowercase :", lowercase)
print('='*80)

print(fiters)