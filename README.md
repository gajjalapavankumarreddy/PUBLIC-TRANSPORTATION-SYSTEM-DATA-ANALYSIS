# PUBLIC-TRANSPORTATION-SYSTEM-DATA-ANALYSIS
This project analyzes APSRTC's public transportation data using Python libraries like Pandas, NumPy, Matplotlib, and Seaborn. It identifies patterns in revenue, fuel efficiency, and occupancy to improve route planning, reduce costs, and enhance commuter experience through data-driven decisions.
### Import the required Libraries and Frameworks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

### Download the dataset from Kaggle and Read it using Pandas
df = pd.read_csv("APSRTC_Transport_Data.csv")
df.head()
### Data Exploration
df.info
df.describe()
### Data Cleaning 
#### Missing values:
df.isnull().sum()
### Data Grouping and Aggregation using Pandas
df['revenue_per_km'] = df['revenue'] / df['distance_km']
df['fuel_efficiency_kmpl'] = df['distance_km'] / df['fuel_consumed_liters']
df['passengers_per_km'] = df['passengers'] / df['distance_km']
df['occupancy_category'] = pd.cut(df['occupancy_rate'], 
                                 bins=[0, 50, 70, 85, 100], 
                                 labels=['Low', 'Medium', 'High', 'Very High'])
df.head()
route_analysis = df.groupby('route').agg({
    'passengers': ['sum', 'mean'],
    'revenue': ['sum', 'mean'],
    'occupancy_rate': 'mean',
    'distance_km': 'mean'
}).round(2)
route_analysis
bus_type_analysis = df.groupby('bus_type').agg({
    'passengers': ['sum', 'mean'],
    'revenue': ['sum', 'mean'],
    'occupancy_rate': 'mean',
    'fare_per_passenger': 'mean'
}).round(2)
bus_type_analysis 
### Data Visualisation
#### 1. Revenue by Bus Type
bus_revenue = df.groupby('bus_type')['revenue'].sum().sort_values(ascending=False)
plt.figure(figsize=(14, 8))  
bars = plt.bar(bus_revenue.index, bus_revenue.values, color='#1f77b4', edgecolor='black')
plt.title('Total Revenue by Bus Type', fontsize=22, fontweight='bold')
plt.xlabel('Bus Type', fontsize=16)
plt.ylabel('Total Revenue (₹)', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + yval*0.01, f'₹{yval:,.0f}', 
             ha='center', va='bottom', fontsize=13, fontweight='medium')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#### 2. Occupancy Rate Distribution
plt.figure(figsize=(14, 8)) 
plt.hist(df['occupancy_rate'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(df['occupancy_rate'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {df["occupancy_rate"].mean():.1f}%')
plt.title('Distribution of Occupancy Rates', fontsize=20, fontweight='bold')
plt.xlabel('Occupancy Rate (%)', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#### 3. Passengers by Route
plt.figure(figsize=(20, 12))  
plt.subplot(4, 3, 3)
route_passengers = df.groupby('route')['passengers'].sum().sort_values(ascending=True)
plt.barh(route_passengers.index, route_passengers.values, color='coral')
plt.title('Total Passengers by Route', fontsize=18, fontweight='bold')
plt.xlabel('Total Passengers', fontsize=15)
plt.ylabel('Route', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

#### 4. Monthly Revenue Trend
plt.figure(figsize=(16, 10))
monthly_revenue = df.groupby('month')['revenue'].sum()
months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']
monthly_revenue = monthly_revenue.reindex([m for m in months_order if m in monthly_revenue.index])
plt.plot(monthly_revenue.index, monthly_revenue.values, 
         marker='o', color='royalblue', linewidth=3, markersize=10)
plt.title('Monthly Revenue Trend', fontsize=22, fontweight='bold')
plt.xlabel('Month', fontsize=16)
plt.ylabel('Revenue (₹)', fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
for i, value in enumerate(monthly_revenue.values):
    plt.text(i, value + max(monthly_revenue.values)*0.015, f'₹{value:,.0f}', 
             ha='center', fontsize=12)
plt.tight_layout()
plt.show()
#### 5. Fuel Efficiency by Bus Type
plt.figure(figsize=(20, 12))  
plt.subplot(4, 3, 5)
sns.boxplot(data=df, x='bus_type', y='fuel_efficiency_kmpl', palette='Set2', linewidth=2)
plt.title('Fuel Efficiency by Bus Type', fontsize=18, fontweight='bold')
plt.xlabel('Bus Type', fontsize=15)
plt.ylabel('Fuel Efficiency (km/l)', fontsize=15)
plt.xticks(rotation=45, fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
#### 6. Occupancy Rate by Day of Week
plt.figure(figsize=(14, 8))
day_occupancy = df.groupby('day_of_week')['occupancy_rate'].mean()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_occupancy = day_occupancy.reindex([d for d in days_order if d in day_occupancy.index])
bars = plt.bar(day_occupancy.index, day_occupancy.values, color='lightblue', edgecolor='black')
plt.title('Average Occupancy Rate by Day of Week', fontsize=22, fontweight='bold')
plt.xlabel('Day of Week', fontsize=16)
plt.ylabel('Occupancy Rate (%)', fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}%', 
             ha='center', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
#### 7. Revenue vs Distance Scatter Plot
plt.figure(figsize=(20, 12))  # Big canvas for all 12 plots
plt.subplot(4, 3, 7)
scatter = plt.scatter(
    df['distance_km'], 
    df['revenue'], 
    alpha=0.7, 
    c=df['passengers'], 
    cmap='viridis', 
    edgecolor='black',
    s=60  # marker size
)
cbar = plt.colorbar(scatter)
cbar.set_label('Passengers', fontsize=13)
plt.title('Revenue vs Distance (colored by passengers)', fontsize=18, fontweight='bold')
plt.xlabel('Distance (km)', fontsize=15)
plt.ylabel('Revenue (₹)', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

#### 8. Depot Performance
plt.figure(figsize=(16, 10))
depot_performance = df.groupby('depot').agg({
    'revenue': 'sum',
    'passengers': 'sum'
}).sort_values('revenue', ascending=False)
bars = plt.bar(depot_performance.index, depot_performance['revenue'], 
               color='gold', edgecolor='black')
plt.title('Revenue by Depot', fontsize=22, fontweight='bold')
plt.xlabel('Depot', fontsize=16)
plt.ylabel('Revenue (₹)', fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.4)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + max(depot_performance['revenue'])*0.01, 
             f'₹{height:,.0f}', ha='center', fontsize=12)
plt.tight_layout()
plt.show()
#### 9. Occupancy Category Distribution
plt.figure(figsize=(10, 10)) 
occupancy_dist = df['occupancy_category'].value_counts()
plt.pie(
    occupancy_dist.values, 
    labels=occupancy_dist.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=plt.cm.Set3.colors, 
    wedgeprops={'edgecolor': 'black', 'linewidth': 1}
)
plt.title('Distribution of Occupancy Categories', fontsize=22, fontweight='bold')
plt.tight_layout()
plt.show()

#### 10. Correlation Heatmap
plt.figure(figsize=(12, 10)) 

correlation_cols = ['passengers', 'occupancy_rate', 'distance_km', 
                    'revenue', 'fuel_consumed_liters', 'fuel_efficiency_kmpl']
correlation_matrix = df[correlation_cols].corr()
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt='.2f',
    linewidths=0.5,
    linecolor='gray',
    square=True,
    annot_kws={"size": 12}
)
plt.title('Correlation Matrix of Key Operational Metrics', fontsize=22, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.show()

#### 11. Revenue Distribution by Bus Type
plt.figure(figsize=(14, 10))
sns.violinplot(
    data=df,
    x='bus_type',
    y='revenue',
    inner='box',            
    palette='muted',
    scale='area',           
    linewidth=1
)
plt.title('Revenue Distribution by Bus Type', fontsize=22, fontweight='bold')
plt.xlabel('Bus Type', fontsize=16)
plt.ylabel('Revenue (₹)', fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

#### 12. Time Series of Daily Revenue
plt.figure(figsize=(16, 10))
daily_revenue = df.groupby('date')['revenue'].sum().sort_index()
plt.plot(
    daily_revenue.index, 
    daily_revenue.values, 
    color='royalblue', 
    linewidth=2.5, 
    marker='o', 
    markersize=5, 
    alpha=0.9
)
plt.title('Daily Revenue Time Series', fontsize=24, fontweight='bold')
plt.xlabel('Date', fontsize=16)
plt.ylabel('Revenue (₹)', fontsize=16)
plt.xticks(rotation=45, fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True, linestyle='--', alpha=0.4)
rolling_avg = daily_revenue.rolling(window=7).mean()
plt.plot(daily_revenue.index, rolling_avg, color='red', linestyle='--', label='7-Day Avg', linewidth=2)
plt.legend()
plt.tight_layout()
plt.show()

### Advanced Analytics and Key Insights
#### Top 5 Routes by Revenue
print("Top 5 Routes by Revenue:")
top_routes = df.groupby('route')['revenue'].sum().sort_values(ascending=False).head()
for route, revenue in top_routes.items():
    print(f"  {route}: ₹{revenue:,.0f}")
#### Best performing bus types
bus_performance = df.groupby('bus_type').agg({
    'revenue': 'sum',
    'passengers': 'sum',
    'occupancy_rate': 'mean'
}).sort_values('revenue', ascending=False)
print(bus_performance)
#### Efficiency metrics
print(f"Average Occupancy Rate: {df['occupancy_rate'].mean():.2f}%")
print(f"Average Fuel Efficiency: {df['fuel_efficiency_kmpl'].mean():.2f} km/l")
print(f"Average Revenue per KM: ₹{df['revenue_per_km'].mean():.2f}")
#### Seasonal analysis
seasonal_analysis = df.groupby('month').agg({
    'passengers': 'sum',
    'revenue': 'sum',
    'occupancy_rate': 'mean'
}).round(2)
print(seasonal_analysis)
#### Create summary statistics
print(f"Total Records Analyzed: {len(df):,}")
print(f"Total Revenue Generated: ₹{df['revenue'].sum():,.0f}")
print(f"Total Passengers Served: {df['passengers'].sum():,}")
print(f"Total Distance Covered: {df['distance_km'].sum():,.0f} km")
print(f"Total Fuel Consumed: {df['fuel_consumed_liters'].sum():,.0f} liters")
print(f"Average Occupancy Rate: {df['occupancy_rate'].mean():.2f}%")

#### Export data summary
print("Dataset is ready for further analysis or export")
print(f"Columns: {list(df.columns)}")
print(f"Data types: {df.dtypes.to_dict()}")
#### sample of final processed data
print("\nSample of processed data:")
print(df[['bus_id', 'route', 'bus_type', 'passengers', 'revenue', 'occupancy_rate', 'fuel_efficiency_kmpl']].head())

