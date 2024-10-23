import time
from functools import wraps

# Decorator to log actions
def log_action(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"Action: {func.__name__} | Timestamp: {timestamp}")
        return func(*args, **kwargs)
    return wrapper

# Example functions representing user actions
@log_action
def login(username):
    print(f"{username} logged in successfully.")

@log_action
def update_profile(username, new_email):
    print(f"{username} updated their profile. New email: {new_email}")

@log_action
def make_purchase(username, item):
    print(f"{username} purchased {item}.")

# Test the decorated functions
login("johndoe")
update_profile("johndoe", "john@example.com")
make_purchase("johndoe", "laptop")
