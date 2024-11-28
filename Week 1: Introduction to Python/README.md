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
phone_number: str ...
email: ...

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
contacts = ["John Doe", ...]

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
# List of contacts. Each contact is a dictionary with keys 'name', 'phone_number', and 'email'
contacts = []

def add_contact(name: str, phone_number: str, email: str) -> int:
    ...
    # contact_id is the index of the contact in the contacts list
    return contact_id

def update_contact(id: int, name: str, phone_number: str, email: str) -> None:
    ...

def display_contacts():
    ...

# Test the functions
contact_id = add_contact("John Doe", "123-456-7890", "john@example.com")
_ = add_contact("Alice Smith", "123-456-7777", "alice@example.com")
update_contact(
    id=contact_id,
    name="John Doe",
    phone_number="555-555-5555",
    email="john@example.com"
)
display_contacts()
```

## Expected Output:
```
Contact Information (ID: 0):
Name: John Doe
Phone Number: 555-555-5555
Email: john@example.com

Contact Information (ID: 1):
Name: Alice Smith
Phone Number: 123-456-7777
Email: alice@example.com
```

# Task 4: JSON

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
}

# Save book information to a JSON file
with open("book.json", "w") as file:
    json.dump(book, file)

# Load book information from the JSON file
with open("book.json", "r") as file:
    book_info = json.load(file)

print(f"""
    Book Information:
    Title: {book_info["title"]}
    Author: {book_info["author"]}
    Year: {book_info["year"]}
    ISBN: {book_info["ISBN"]}
    """)
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

# Task 7: Installing and Using External Libraries

In python, the standard package manager is `pip`. You can use `pip` to install external libraries that are not part of the standard library. In this task, you will install the `requests` library and use it to make a simple HTTP GET request to a public API. The `requests` library is a popular library for making HTTP requests in Python and does not come pre-installed with Python. 

We will install the package into a virtual environment, using `venv`. Virtual environments help us separate the dependencies for different projects and is a best practice in Python development.

## Tasks:
- Create a virtual environment using `venv`.
- Activate the virtual environment.
- Install the `requests` library using `pip`.
- Use the `requests` library to make a GET request to the URL `https://jsonplaceholder.typicode.com/posts/1` and display the response content.

## Starting Point:
If you are using MacOS or Linux, you can create a virtual environment using the following commands:
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

If you are using Windows, you can create a virtual environment using the following commands:
```cmd
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

After activating the virtual environment, you can install the `requests` library using `pip`:
```bash
pip install requests
```

Now you can use the `requests` library to make an HTTP GET request:
```python
import requests

response = requests.get("https://jsonplaceholder.typicode.com/posts/1")
print(response.content)
```

To reproduce the environment, you can create a `requirements.txt` file using the following command:
```bash
pip freeze > requirements.txt
```

You can install the dependencies from the `requirements.txt` file using the following command:
```bash
pip install -r requirements.txt
```

Lastly, to deactivate the virtual environment, you can use the following command:
```bash
deactivate # For both MacOS/Linux and Windows
```

## Expected Output:
The output should be the content of the response from the API, which is a JSON object representing a post. It should look something like this:
```python
b'{\n  "userId": 1,\n  "id": 1,\n  "title": "sunt aut facere repellat provident occaecati excepturi optio reprehenderit",\n  "body": "quia et suscipit\\nsuscipit recusandae consequuntur expedita et cum\\nreprehenderit molestiae ut ut quas totam\\nnostrum rerum est autem sunt rem eveniet architecto"\n}'
```

# Task 8: Python dictionaries vs Dataclasses

Activate the virtual environment again you created in Task 7. We will now decode the JSON response from the API into a Python dataclass. Dataclasses are a typed alternative to dictionaries (`{}`). Dataclasses provide extend the functionality of dictionaries by allowing you to make the dictionary conform to a schema. Dataclasses are also a class which means you can add methods to them unlike to dictionaries.

## Tasks:
- Create a dataclass `Post` with attributes `userId`, `id`, `title`, and `body`.
- Decode the JSON response from the API into an instance of the `Post` dataclass. Repeat for 3 posts.

## Starting Point:
```python
import requests
import json

from dataclasses import dataclass

@dataclass
class Post:
    userId: int
    ...


posts = []
# Decode the JSON response into a Post instance
for ...
    # URL: https://jsonplaceholder.typicode.com/posts/{post_id}
    # post_id is a number from 1 to max number of posts
    response = requests.get(...)
    post_data = json.loads(response.content)

    post = Post(userId=post_data["userId"], ...)
    posts.append(post)

# NOTE: post.userId to access the userId of the post
...
```

## Expected Output:
```
Post 1:
    User ID: 1
    ID: 3
    Title: ea molestias quasi exercitationem repellat qui ipsa sit aut
    Body: et iusto sed quo iure
voluptatem occaecati omnis eligendi aut ad
voluptatem doloribus vel accusantium quis pariatur
molestiae porro eius odio et labore et velit aut

Post 2:
    User ID: 1
    ID: 2
    Title: qui est esse
    Body: est rerum tempore vitae
sequi sint nihil reprehenderit dolor beatae ea dolores neque
fugiat blanditiis voluptate porro vel nihil molestiae ut reiciendis
qui aperiam non debitis possimus qui neque nisi nulla

Post 3:
    User ID: 1
    ID: 3
    Title: ea molestias quasi exercitationem repellat qui ipsa sit aut
    Body: et iusto sed quo iure
voluptatem occaecati omnis eligendi aut ad
voluptatem doloribus vel accusantium quis pariatur
molestias et enim adipisci aut delectus
```