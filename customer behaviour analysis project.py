#customer behaviour analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
df = pd.read_csv(r"C:\Users\boser\Documents\customer_csv.csv")
if 'age' in df.columns:
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
for col in ['age', 'purchase_amount', 'income']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
if 'order_date' in df.columns:
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
df = df.dropna(subset=['id', 'purchase_amount'])
customer_spend = df.groupby('id')['purchase_amount'].sum().reset_index()
plt.figure(figsize=(8,5))
sns.histplot(customer_spend['purchase_amount'], bins=50, kde=True)
plt.title("Distribution of Customer Spend")
plt.xlabel("Total Spend")
plt.ylabel("Number of Customers")
plt.show()
top_products = df['Products'].value_counts().head(10)
print("\nTop 10 Products:\n", top_products)
plt.figure(figsize=(8,5))
sns.barplot(x=top_products.values, y=top_products.index, palette="viridis")
plt.title("Top 10 Most Purchased Products")
plt.xlabel("Purchase Count")
plt.ylabel("Products")
plt.show()
X = customer_spend[['purchase_amount']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customer_spend['Segment'] = kmeans.fit_predict(X)
segment_means = customer_spend.groupby('Segment')['purchase_amount'].mean().sort_values()
segment_map = {segment_means.index[0]: "Low Value",
               segment_means.index[1]: "Medium Value",
               segment_means.index[2]: "High Value"}
customer_spend['Segment'] = customer_spend['Segment'].map(segment_map)
print("\nCustomer Segmentation Sample:\n", customer_spend.head())
segment_revenue = customer_spend.groupby('Segment')['purchase_amount'].sum().sort_values(ascending=False)
print("\nRevenue Contribution by Segment:\n", segment_revenue)
plt.figure(figsize=(6,5))
segment_revenue.plot(kind='bar', color=['#ff9999','#66b3ff','#99ff99'])
plt.title("Revenue Contribution by Segment")
plt.ylabel("Total Revenue")
plt.show()
df['Month'] = df['order_date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['purchase_amount'].sum()
plt.figure(figsize=(10,5))
monthly_sales.index = monthly_sales.index.astype(str)  
monthly_sales.plot(kind='line', marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Spend")
plt.xticks(rotation=45)
plt.show()