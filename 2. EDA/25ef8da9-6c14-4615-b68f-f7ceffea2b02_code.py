# In[3]
import pandas as pd 
order_file = '/datasets/instacart_orders.csv'
products_file = '/datasets/products.csv'
aisles_file = '/datasets/aisles.csv'
departments_file = '/datasets/departments.csv'
order_products_file = '/datasets/order_products.csv'

# In[4]
orders_df = pd.read_csv(order_file, sep=';', low_memory=False)
print(orders_df)
print(orders_df.info())

# In[5]
products_df = pd.read_csv(products_file, sep=';', low_memory=False)
print(products_df)
print(products_df.info())

# In[6]
aisles_df = pd.read_csv(aisles_file, sep=';', low_memory=False)
print(aisles_df)
print(aisles_df.info())

# In[7]
departments_df = pd.read_csv(departments_file, sep=';', low_memory=False)
print(departments_df)
print(departments_df.info())

# In[8]
order_products_df = pd.read_csv(order_products_file, sep=';', low_memory=False)
print(order_products_df)
print(order_products_df.info())

# In[9]
orders_df.fillna({'order_number': 0, 'order_dow': 0, 'order_hour_of_day': 0}, inplace=True)
order_products_df.fillna({'add_to_cart_order': 0}, inplace=True)

# In[12]
# Check for duplicated orders
duplicated_orders = orders_df[orders_df.duplicated()]
print("Duplicated Orders:")
print(duplicated_orders)

# In[13]
# Check for all orders placed Wednesday at 2:00 AM
wednesday_orders_2am = orders_df[(orders_df['order_dow'] == 3) & (orders_df['order_hour_of_day'] == 2)]
print("Orders placed Wednesday at 2:00 AM:")
print(wednesday_orders_2am)

# In[14]
# Remove duplicate orders
orders_df.drop_duplicates(inplace=True)
print("Removed duplicate orders")
print(orders_df.duplicated().sum())

# In[15]
# Double check for duplicate rows
print("Double check for duplicate rows")
print(orders_df.duplicated().sum())

# In[16]
# Double check for duplicate order IDs only
duplicate_order_ids = orders_df[orders_df.duplicated(subset=['order_id'])]
print("Duplicate Order IDs")
print(duplicate_order_ids)

# In[19]
# Check for fully duplicate rows
duplicated_products = products_df[products_df.duplicated()]
print("Fully Duplicate Products:")
print(duplicated_products)

# In[20]
# Check for just duplicate product IDs
duplicate_product_ids = products_df[products_df.duplicated(subset=['product_id'])]
print("Duplicate Product IDs")
print(duplicate_product_ids)

# In[21]
# Check for just duplicate product names (convert names to lowercase to compare better)
products_df['product_name_lower'] = products_df['product_name'].str.lower()
duplicate_product_names = products_df[products_df.duplicated(subset=['product_name_lower'])]
print("Duplicate Product Names:")
print(duplicate_product_names)

# In[22]
# Check for duplicate product names that aren't missing
non_missing_duplicate_product_names = products_df[products_df['product_name'].notnull() & products_df.duplicated(subset=['product_name_lower'])]
print("Non-missing Duplicate Product Names:")
print(non_missing_duplicate_product_names)

# In[25]
duplicated_departments = departments_df[departments_df.duplicated()]
print("Fully Duplicate Departments:")
print(duplicated_departments)

# In[26]
duplicate_department_ids = departments_df[departments_df.duplicated(subset=['department_id'])]
print("Duplicate Department IDs:")
print(duplicate_department_ids)

# In[29]
duplicated_aisles = aisles_df[aisles_df.duplicated()]
print("Fully Duplicate Aisles:")
print(duplicated_aisles)

# In[30]
duplicate_aisle_ids = aisles_df[aisles_df.duplicated(subset=['aisle_id'])]
print("Duplicate Aisle IDs:")
print(duplicate_aisle_ids)

# In[33]
# Check for fullly duplicate rows
duplicated_order_products = order_products_df[order_products_df.duplicated()]
print("Fully Duplicate Order Products:")
print(duplicated_order_products)

# In[34]
# Double check for any other tricky duplicates
tricky_duplicates = order_products_df[order_products_df.duplicated(subset=['order_id', 'product_id'])]
print("Tricky Duplicates(based on order_id and product_id):")
print(tricky_duplicates)

# In[38]
missing_product_names = products_df[products_df['product_name'].isnull()]
print("Missing Product Names:")
print(missing_product_names)

# In[39]
# Are all of the missing product names associated with aisle ID 100?
missing_in_aisle_100 = missing_product_names['aisle_id'].eq(100).all()
print('Are all missing product names associated with aisle ID 100?')
print(missing_in_aisle_100)

# In[40]
# Are all of the missing product names associated with department ID 21?
missing_in_department_21 = missing_product_names['department_id'].eq(21).all()
print("Are all missing product names associated with department ID 21")
print(missing_in_department_21)

# In[41]
# What is this ailse and department?
missing_product_aisles_departments = missing_product_names[['aisle_id', 'department_id']]
print("Aisle and Department for Missing Product Names:")
print(missing_product_aisles_departments)

# In[42]
# Fill missing product names with 'Unknown'
products_df['product_name'].fillna('Unknown', inplace=True)

# In[45]
print(orders_df)
print(orders_df.info())

# In[46]
# Are there any missing values where it's not a customer's first order?
missing_values_not_first_order = orders_df[(orders_df['user_id'].duplicated(keep='first')) & (orders_df.isnull().any(axis=1))]
print("Missing Values ofr Non-First Orders:")
print(missing_values_not_first_order)

# In[49]
print(order_products_df)
print(order_products_df.info())

# In[50]
# What are the min and max values in this column?
min_add_to_cart_order = order_products_df['add_to_cart_order'].min()
max_add_to_cart_order = order_products_df['add_to_cart_order'].max()
print("Min add to cart order:")
print(min_add_to_cart_order)
print("Max add to cart order:")
print(max_add_to_cart_order)

# In[51]
# Save all order IDs with at least one missing value in 'add_to_cart_order'
order_id_with_missing_value = order_products_df[order_products_df['add_to_cart_order'].isnull()] ['order_id'].unique()
print("Order IDs with missing value")
print(order_id_with_missing_value)

# In[52]
# Do all orders with missing values have more than 64 products?
missing_orders_check = order_products_df[order_products_df['add_to_cart_order'].isnull()] ['order_id'].value_counts() 
all_have_more_than_64 = (missing_orders_check > 64).all()
print("Orders with missing value more than 64 products")
print(all_have_more_than_64)

# In[53]
# Replace missing values with 999 and convert column to integer type
order_products_df['add_to_cart_order'].fillna(999, inplace=True)
order_products_df['add_to_cart_order'] = order_products_df['add_to_cart_order'].astype(int)
print(order_products_df['add_to_cart_order'])

# In[57]
valid_hour_range = orders_df['order_hour_of_day'].between(0, 23).all()

print("Are all order_hour_of_day values valid?")
print(valid_hour_range)

# In[58]
valid_dow_range = orders_df['order_dow'].between(0, 6).all()
print("Are all order_dow values valid?")
print(valid_dow_range)

# In[60]
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
orders_df['order_hour_of_day'].hist(bins=24)
plt.title('Distribution of Orders by Hours of Day')
plt.xlabel('Hours of Day')
plt.ylabel('Number of Orders')
plt.xticks(range(0, 24))
plt.grid(axis='y')
plt.show()

print("Most people shop for groceries between the hours of 9:30 AM and 10:30 AM")

# In[64]
plt.figure(figsize=(10,6))
orders_df['order_dow'].hist(bins=7)
plt.title('Distribution of Orders by Day of Week')
plt.xlabel('Day of Week(0=Sun, 6=Sat)')
plt.ylabel('Number of Orders')
plt.xticks(range(0, 7))
plt.grid(axis='y')
plt.show()
print("Most people shop for groceries on Sunday")

# In[68]
#calculate the time between orders
orders_df['days_since_prior_order'] = pd.to_numeric(orders_df['days_since_prior_order'], errors='coerce')


#sort by user_id and order_time
orders_df.sort_values(by=['user_id', 'order_number'], inplace=True)

#calculate the time difference between consecutive orders
orders_df['time_diff'] = orders_df['days_since_prior_order']

# Remove NaN values for plotting
filtered_df = orders_df.dropna(subset=['time_diff'])

#Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(filtered_df['time_diff'], bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Time Between Orders (days)')
plt.ylabel('Frequency')
plt.title('Distribution of Time Between Orders')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Conclusion
avg_time_between_orders = filtered_df['time_diff'].mean()
print(f"On average, customers wait approximately {avg_time_between_orders:.2f} days before placing another order.")

# In[73]
wednesday_orders = orders_df[orders_df['order_dow'] == 3]
saturday_orders = orders_df[orders_df['order_dow'] == 6]

plt.hist(wednesday_orders['order_hour_of_day'], bins=24, alpha=0.5, label='Wednesday')
plt.hist(saturday_orders['order_hour_of_day'], bins=24, alpha=0.5, label='Saturday')

plt.title('Order Hour of Day Distribution: Wednesday vs Saturday')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Orders')
plt.xticks(range(0, 24))
plt.legend()
plt.grid(axis='y')

plt.show()

# In[77]
orders_per_customer = orders_df['user_id'].value_counts()
print(orders_per_customer)

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(orders_per_customer, bins=30, edgecolor='black', alpha=0.7)

# Labels and title
plt.title('Distribution of Number of Orders per Customer')
plt.xlabel('Number of Orders')
plt.ylabel('Number of Customers')
plt.grid(axis='y')
plt.show()

print("Most Customers Have a Low Number of Orders. The distribution is right-skewed, meaning the majority of customers place only a few orders, while a smaller group of frequent shoppers place a high number of orders. A Few Customers Are Highly ActiveSome customers place 20+ orders, indicating a loyal user base that regularly uses the service")

# In[81]
top_products = order_products_df['product_id'].value_counts().head(20).reset_index()
top_products.columns = ['product_id', 'order_count']

# In[82]
top_product_names = products_df[['product_id', 'product_name']].drop_duplicates()
top_products_info = top_products.merge(top_product_names, on='product_id', how='left')

# In[83]
top_products_info = top_products_info[['product_id', 'product_name', 'order_count']]
print("Top 20 Popular Products")
print(top_products_info)

# In[88]
items_per_order = order_products_df.groupby('order_id')['product_id'].count()
print(items_per_order)

print("The typical order size is {items_per_order.mean():.2f}")

# In[89]
plt.figure(figsize=(10,6))
items_per_order.hist(bins=50)
plt.title('Distribution of Number of Items per Order')
plt.xlabel('Number of Items')
plt.ylabel('Number of Orders')
plt.grid(axis='y')
plt.show()

# In[93]
reordered_items = (order_products_df[order_products_df['reordered'] == 1]['product_id'].value_counts().head(20).reset_index())
print(reordered_items)

# In[94]
reordered_items.columns = ['product_id', 'reorder_count'] 
reordered_product_names = products_df[['product_id', 'product_name']].drop_duplicates()

print(reordered_product_names)

# In[95]
reordered_products_info = reordered_items.merge(reordered_product_names, on='product_id', how='left')
reordered_products_info = reordered_products_info[['product_id', 'product_name', 'reorder_count']]

print("Top 20 Reordered Products:")
print(reordered_products_info)

# In[98]
total_orders_per_product = order_products_df['product_id'].value_counts().reset_index()
total_orders_per_product.columns = ['product_id', 'total_orders']

print(total_orders_per_product)

# In[99]
reordered_per_product = order_products_df[order_products_df['reordered'] == 1]['product_id'].value_counts().reset_index()
reordered_per_product.columns = ['product_id', 'num_reordered']

print(reordered_per_product)

# In[100]
product_reorder_ratio = total_orders_per_product.merge(reordered_per_product, on='product_id', how='left')
product_reorder_ratio['num_reordered'].fillna(0, inplace=True)
product_reorder_ratio['reorder_proportion'] = product_reorder_ratio['num_reordered'] / product_reorder_ratio['total_orders']
product_reorder_ratio = product_reorder_ratio.merge(products_df[['product_id', 'product_name']], on='product_id', how='left')
product_reorder_ratio = product_reorder_ratio[['product_id', 'product_name', 'total_orders', 'num_reordered', 'reorder_proportion']]

print("Proportion of Orders That Are Reorders for Each Product:")
print(product_reorder_ratio)

# In[102]
order_products_df = order_products_df.merge(orders_df[['order_id', 'user_id']], on='order_id', how='left')

total_products_per_customer = order_products_df.groupby('user_id')['product_id'].count().reset_index()
total_products_per_customer.columns = ['user_id', 'total_products']

reordered_per_customer = order_products_df[order_products_df['reordered'] == 1].groupby('user_id')['product_id'].count().reset_index()
reordered_per_customer.columns = ['user_id', 'num_reordered']

customer_reorder_ratio = total_products_per_customer.merge(reordered_per_customer, on='user_id', how='left')
customer_reorder_ratio['num_reordered'].fillna(0, inplace=True)

customer_reorder_ratio['reorder_proportion'] = customer_reorder_ratio['num_reordered'] / customer_reorder_ratio['total_products']

print("Proportion of Reordered Products for Each Customer:")
print(customer_reorder_ratio.head())

# In[104]
# Identify the first items added to the cart (order_id irrelevant now)
first_items_count = order_products_df['product_id'].value_counts().reset_index()
first_items_count.columns = ['product_id', 'first_item_count']

# Get product names from products DataFrame
first_items_product_names = products_df[['product_id', 'product_name']]

# Merge counts with product names
first_items_info = first_items_count.merge(first_items_product_names, on='product_id', how='left')

# Ensure all product names are filled
first_items_info['product_name'].fillna("Unknown Product", inplace=True)

# Define a function to clean and group similar product names
def standardize_product_name(name):
    name = name.lower().strip()  # Convert to lowercase and strip spaces
    
    # Remove unnecessary words
    remove_words = ["organic", "bag of", "fat free", "reduced fat", "whole", "2%", "1%", "half & half", "skim", "low fat"]
    for word in remove_words:
        name = name.replace(word, "")

    # Singularize common plural words
    singular_map = {
        "bananas": "banana",
        "avocados": "avocado",
        "grapefruits": "grapefruit",
        "onions": "onion",
        "strawberries": "strawberry",
        "raspberries": "raspberry",
        "blueberries": "blueberry",
        "limes": "lime",
        "lemons": "lemon",
        "waters": "water",
        "apples": "apple",
        "tomatoes": "tomato"
    }
    for plural, singular in singular_map.items():
        name = name.replace(plural, singular)

    # Clean extra spaces
    name = " ".join(name.split()).strip()
    
    return name

# Apply standardization
first_items_info['cleaned_product_name'] = first_items_info['product_name'].apply(standardize_product_name)

# Group by standardized names and sum the counts
grouped_first_items = (
    first_items_info.groupby(['cleaned_product_name'], as_index=False)
    .agg({'first_item_count': 'sum'})
    .sort_values(by='first_item_count', ascending=False)
)

# Ensure empty names are properly labeled
grouped_first_items = grouped_first_items[grouped_first_items['cleaned_product_name'] != ""]

# Merge back with product_id (keeping the most popular one for reference)
first_product_ids = first_items_info.groupby('cleaned_product_name')['product_id'].first().reset_index()
top_20_first_items = grouped_first_items.merge(first_product_ids, on='cleaned_product_name', how='left')

# Select top 20 products
top_20_first_items = top_20_first_items[['product_id', 'cleaned_product_name', 'first_item_count']].head(20)

# Add index column
top_20_first_items.reset_index(drop=True, inplace=True)
top_20_first_items.index += 1  # Start index from 1

# Display the result
print("Top 20 Items That People Put in Their Carts First:")
print(top_20_first_items)

