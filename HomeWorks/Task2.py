# Task2
def operations():
    add = lambda x, y: x + y
    subtract = lambda x, y: x - y
    multiply = lambda x, y: x * y
    divide = lambda x, y: x / y if y != 0 else "Error"

    while True:
        choice = input("Number (1,2,3,4,5): ")
        if choice == '5':
            print("Exit")
            break
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))
        if choice == '1':
            print("Result:", add(num1, num2))
        elif choice == '2':
            print("Result:", subtract(num1, num2))
        elif choice == '3':
            print("Result:", multiply(num1, num2))
        elif choice == '4':
            print("Result:", divide(num1, num2))
        else:
            print("Choice")

operations()
