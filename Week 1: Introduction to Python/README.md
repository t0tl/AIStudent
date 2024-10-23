# Task 1: Variables and Data Types
## Story:
You are building a simple program to manage a list of contacts. Each contact has a name, phone number, and email address. You need to create variables to store this information and display it in a readable format.

## Tasks:
- Create variables to store the name, age, phone number, and email address of a contact.
- Display the contact information in a readable format using f-strings.

## Starting Point:
```python
# Contact information
name: str = ...
age: int = ...
phone_number: str = ...
email: str = ...

# Display contact information
print("Contact Information:")
print(f"Name: {name}")
...
```

## Expected Output:
```
Name: John Doe
Phone Number: 123-456-7890
Email: john@example.com
```

# Task 2: Loops and Conditionals
## Story:
You are building a simple program to manage a list of contacts. You want to display a welcome message for each contact based on their name. If the name starts with 'A', you want to display a special message.

## Tasks:
- Create a list of contact names.
- Use a loop to iterate over the list and display a welcome message for each contact.
- If the name starts with 'A', display a special message.

## Starting Point:
```python
# List of contact names
contacts = [...]

# Display welcome message for each contact
# Remember, special message for names starting with 'A'
for name in contacts:
    ...
```

## Expected Output:
```
Welcome John Doe!
Welcome Alice Smith! (Special message for names starting with 'A')
Welcome Bob Johnson!
```

# Task 3: Functions and Modules
## Story:
You are building a simple program to manage a list of contacts. You want to create functions to add a new contact, update an existing contact, and display all contacts.

## Tasks:
- Create functions `add_contact`, `update_contact`, and `display_contacts`.
- Implement the functions to add a new contact, update an existing contact, and display all contacts.
- Test the functions by adding a new contact, updating an existing contact, and displaying all contacts.

## Starting Point:
```python
# List of contacts
contacts = []

def add_contact(name, phone_number, email):
    ...

def update_contact(name, phone_number, email):
    ...

def display_contacts():
    ...

# Test the functions
add_contact("John Doe", "123-456-7890", "john@example.com")
update_contact("John Doe", "555-555-5555", "john@example.com")
display_contacts()
```

## Expected Output:
```
Contact Information:
Name: John Doe
Phone Number: 555-555-5555
Email: john@example.com
```

# Task 4: Dictionaries / JSON

## Story:
You are building a program which stores information about books in a library to disk. You want to use a dictionary to store the book information and save it to a JSON file for persistence.

## Tasks:
- Create a dictionary to store information about a book, including `title`, `author`, `year`, and `ISBN`.
- Save the book information to a JSON file using the `json` module.
- Load the book information from the JSON file and display it.

## Starting Point:
```python
import json

# Book information
book = {
    ...
}

# Save book information to a JSON file
with open("book.json", "w") as file:
    ...

# Load book information from the JSON file
with open("book.json", "r") as file:
    ...
    book_info = ...

print("Book Information:")
print(f"Title: {book_info['title']}")
...
```

## Expected Output:
```
Book Information:
Title: The Great Gatsby
Author: F. Scott Fitzgerald
Year: 1925
ISBN: 9780743273565
```

# Task 5: Car Dealership Inventory
## Story:
You manage the inventory for a car dealership. The dealership sells various types of vehicles like cars and trucks. You need to organize vehicle information, including calculating the total inventory value and displaying information about each vehicle in a readable format.

## Tasks:
- Create a base Vehicle class with attributes for `make`, `model`, `year`, and `price`.
- Create a `Car` and a `Truck` subclass, each with a specific feature. Cars have `num_doors`, and trucks have `payload_capacity`.
- Implement the `__str__` dunder method to display vehicle information neatly.
- Using the class attributes, calculate the total value of the inventory consisting of a 2023 Toyota Camry, worth $24,000, a 2022 Ford F-150, worth $35,000, and a 2021 Honda Civic, worth $22,000. The Ford F-150 has a payload capacity of 1000 kg and the other two vehicles have 4 doors.

## Starting Point:
```python
# Base class Vehicle
class Vehicle:
    ...

# Inventory list
inventory = [
    ...
]

# Display all vehicles and calculate total value
total_value = 0
for vehicle in inventory:
    ...

print(f"Total inventory value: ${total_value}")
```

## Expected Output:
```
2023 Toyota Camry - $24000 (Car, 4 doors)

2022 Ford F-150 - $35000 (Truck, Payload capacity: 1000 kg)

2021 Honda Civic - $22000 (Car, 4 doors)

Total inventory value: $81000
```

# Task 6: Logging and Decorators

## Story:
You are building a user management system that logs user actions like logging in, updating profiles, and making purchases for regulatory reasons. You want to add a decorator to log the timestamp of each action.

## Tasks:
- Create a decorator function `log_action` that logs the action name and timestamp.
- Decorate the `login`, `update_profile`, and `make_purchase` functions with the `log_action` decorator.
- Test the decorated functions by calling them with sample arguments.

## Starting Point:

```python
import time
from functools import wraps

# Decorator to log actions
def log_action(func):
    ...

def login(username):
    print(f"{username} logged in successfully.")

def update_profile(username, new_email):
    print(f"{username} updated their profile. New email: {new_email}")

def make_purchase(username, item):
    print(f"{username} purchased {item}.")

# Test the decorated functions
login("johndoe")
update_profile("johndoe", "john@example.com")
make_purchase("johndoe", "laptop")
```

## Expected Output:
```
Action: login | Timestamp: 2022-01-01 12:00:00
johndoe logged in successfully.
Action: update_profile | Timestamp: 2022-01-01 12:00:01
johndoe updated their profile. New email: john@example.com
Action: make_purchase | Timestamp: 2022-01-01 12:00:02
johndoe purchased laptop.
```