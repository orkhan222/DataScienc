# Task1
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squared_a = list(map(lambda x: x**2, a)) 
print(squared_a)

a_sum = 0
for x in a:
    a_sum += x**2
print(a_sum)
