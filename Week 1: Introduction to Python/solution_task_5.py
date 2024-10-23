# Base class Vehicle
class Vehicle:
    def __init__(self, make, model, year, price):
        self.make = make
        self.model = model
        self.year = year
        self.price = price
    
    def __str__(self):
        return f"{self.year} {self.make} {self.model} - ${self.price}"

# Subclass Car
class Car(Vehicle):
    def __init__(self, make, model, year, price, num_doors):
        super().__init__(make, model, year, price)
        self.num_doors = num_doors
    
    def __str__(self):
        return f"{super().__str__()} (Car, {self.num_doors} doors)"

# Subclass Truck
class Truck(Vehicle):
    def __init__(self, make, model, year, price, payload_capacity):
        super().__init__(make, model, year, price)
        self.payload_capacity = payload_capacity
    
    def __str__(self):
        return f"{super().__str__()} (Truck, {self.payload_capacity} kg payload)"

# Inventory list
inventory = [
    Car('Toyota', 'Camry', 2023, 24000, 4),
    Truck('Ford', 'F-150', 2022, 35000, 2000),
    Car('Honda', 'Civic', 2021, 22000, 4)
]

# Display all vehicles and calculate total value
total_value = 0
for vehicle in inventory:
    print(vehicle + "\n")
    total_value += vehicle.price

print(f"Total inventory value: ${total_value}")
