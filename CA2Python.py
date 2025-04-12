import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency

# Load the dataset
file_path = "C:\\Users\\Vikas\\OneDrive\\Desktop\\PythonProject\\Electric_Vehicle_Population_Data (1).csv"
df = pd.read_csv(file_path)

# Dataset Overview
print("Shape of the dataset:", df.shape)
print("\nColumns:\n", df.columns)
print("\nSample records:")
print(df.describe())
print(df.head(10))
print(df.tail(10))
print("\nActual column names:")
print(df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())

# Cleaning the data
df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')
df['Electric Range'] = pd.to_numeric(df['Electric Range'], errors='coerce')
df.dropna(subset=['Make', 'Model Year', 'Electric Range', 'Electric Vehicle Type', 'City', 'County'], inplace=True)

# Hypothesis Testing
bev = df[df['Electric Vehicle Type'] == 'Battery Electric Vehicle']['Electric Range']
phev = df[df['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle']['Electric Range']

# Ensure sufficient sample sizes for the T-test
if len(bev) > 30 and len(phev) > 30:
    t_stat, p_value = ttest_ind(bev, phev, equal_var=False)
    print(f"T-Test Results: t-statistic = {t_stat}, p-value = {p_value}")
else:
    print("Sample size too small for T-test. Skipping hypothesis testing.")

# Chi-Square Test-->Relationship between EV Type and Make
contingency_table = pd.crosstab(df['Make'], df['Electric Vehicle Type'])
chi2, chi_p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square Test Results: chi2 = {chi2}, p-value = {chi_p}, degrees of freedom = {dof}")

# Outlier Detection-->Electric Range
q1 = df['Electric Range'].quantile(0.25)
q3 = df['Electric Range'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = df[(df['Electric Range'] < lower_bound) | (df['Electric Range'] > upper_bound)]
print(f"Number of outliers in Electric Range: {len(outliers)}")

# Visualizations
# Electric Range Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['Electric Range'], bins=30, kde=True, color='darkgreen', label='Electric Range Distribution')
plt.title("Distribution of Electric Vehicle Ranges", color='darkblue')
plt.xlabel("Electric Range (miles)", color='darkblue')
plt.ylabel("Count", color='darkblue')
plt.legend(title='Legend')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Electric Range Distribution by Top 10 Vehicle Makes
top_makes = df['Make'].value_counts().head(10).index
filtered_df = df[df['Make'].isin(top_makes)]
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_df, x='Make', y='Electric Range', hue='Make', palette='coolwarm', dodge=False)
plt.title("Electric Range Distribution by Top 10 Vehicle Makes", color='darkblue')
plt.xlabel("Make", color='darkblue')
plt.ylabel("Electric Range (miles)", color='darkblue')
plt.xticks(rotation=45, color='darkblue')
plt.legend([], frameon=False)
plt.tight_layout()
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(12, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
plt.title("Correlation Between Numerical Features", color='darkgreen')
plt.tight_layout()
plt.show()

# EV Population Growth by Type Over Model Years
grouped = df.groupby(['Model Year', 'Electric Vehicle Type']).size().reset_index(name='Count')
plt.figure(figsize=(12, 6))
sns.lineplot(data=grouped, x='Model Year', y='Count', hue='Electric Vehicle Type', marker='o', palette='husl')
plt.title("EV Population Growth by Type Over Model Years", color='darkred')
plt.xlabel("Model Year", color='darkred')
plt.ylabel("Number of EVs", color='darkred')
plt.tight_layout()
plt.show()

# Combo Chart: Bar + Line
model_year_data = df.groupby('Model Year').agg({'VIN (1-10)': 'count', 'Electric Range': 'mean'}).rename(columns={'VIN (1-10)': 'EV Count', 'Electric Range': 'Avg Range'}).reset_index()
fig, ax1 = plt.subplots(figsize=(12, 6))
sns.barplot(data=model_year_data, x='Model Year', y='EV Count', ax=ax1, color='darkorange')
ax1.set_ylabel('Number of EVs', color='darkorange')
ax1.tick_params(axis='y', labelcolor='darkorange')
plt.xlabel("Model Year", color='darkblue')

ax2 = ax1.twinx()
sns.lineplot(data=model_year_data, x='Model Year', y='Avg Range', ax=ax2, color='blue', marker='o')
ax2.set_ylabel('Average Electric Range (miles)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title("EV Count and Average Electric Range by Model Year", color='purple')
plt.tight_layout()
plt.show()

# Average Electric Range Over Years by Vehicle Type
plt.figure(figsize=(12, 6))
avg_range_by_type_year = df.groupby(['Model Year', 'Electric Vehicle Type'])['Electric Range'].mean().reset_index()
sns.lineplot(data=avg_range_by_type_year, x='Model Year', y='Electric Range', hue='Electric Vehicle Type', marker='o', palette='Set2')
plt.title("Average Electric Range Over Model Years by Vehicle Type", color='darkgreen')
plt.xlabel("Model Year", color='darkgreen')
plt.ylabel("Average Electric Range (miles)", color='darkgreen')
plt.legend(title='Vehicle Type')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
