import pandas as pd
import random

# Define the base data
data = {
    'Email': [],
    'P01': [],
    'P02': [],
    'P03': [],
    'P04': [],
    'P05': [],
    'P06': [],
    'P07': [],
    'P08': [],
    'P09': [],
    'P10': [],
    'P11': [],
    'P12': [],
    'P13': [],
    'P14': [],
    'P15': [],
    'Searches': []
}

# Function to generate random email addresses
def generate_email():
    domains = ["gmail.com", "yahoo.com", "outlook.com", "connect.np.edu.sg"]
    prefix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
    return f"{prefix}@{random.choice(domains)}"

# Function to generate random order counts
def generate_orders():
    return [random.randint(0, 20) for _ in range(15)]

# Function to generate searches prioritizing all items
def generate_searches(orders):
    # Sort items based on their order counts in descending order
    sorted_items = sorted(range(len(orders)), key=lambda i: orders[i], reverse=True)
    return ",".join([f"P{str(item+1).zfill(2)}" for item in sorted_items])

# Generate 200,000 rows of dummy data
for _ in range(200000):
    email = generate_email()
    orders = generate_orders()
    searches = generate_searches(orders)

    data['Email'].append(email)
    for i in range(15):
        data[f'P{str(i+1).zfill(2)}'].append(orders[i])
    data['Searches'].append(searches)

# Create DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv('dummy_data_with_prioritized_items.csv', index=False)
