import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the weather data into a Pandas DataFrame
weather_data = pd.read_csv('seattle-weather.csv')

# Preprocessing
# Clean the data to remove any missing or duplicate values
weather_data.dropna(inplace=True)
weather_data.drop_duplicates(inplace=True)

# Handle any outliers or inconsistencies in the data
weather_data['temp_max'] = np.where(weather_data['temp_max'] < -50, np.nan, weather_data['temp_max'])
weather_data['temp_min'] = np.where(weather_data['temp_min'] < -50, np.nan, weather_data['temp_min'])

# Convert the date column to a datetime format
weather_data['date'] = pd.to_datetime(weather_data['date'])

# Add features to the data, such as month and year
weather_data['year'] = weather_data['date'].dt.year
weather_data['month'] = weather_data['date'].dt.month
weather_data['day'] = weather_data['date'].dt.day

# Drop rows with missing values
weather_data.dropna(inplace=True)

# Visualization
# Create a line plot of maximum and minimum temperatures over time
plt.figure()
sns.lineplot(x='date', y='temp_max', data=weather_data, label='Max Temp')
sns.lineplot(x='date', y='temp_min', data=weather_data, label='Min Temp')
plt.title('Temperature over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show(block=False)

# Create a scatter plot of maximum temperature vs. wind speed
plt.figure()
sns.scatterplot(x='temp_max', y='wind', data=weather_data)
plt.title('Max Temperature vs. Wind Speed')
plt.xlabel('Maximum Temperature (°C)')
plt.ylabel('Wind Speed (m/s)')
plt.show(block=False)

# Create a histogram of maximum temperature distribution
plt.figure()
sns.histplot(x='temp_max', data=weather_data)
plt.title('Max Temperature Distribution')
plt.xlabel('Maximum Temperature (°C)')
plt.ylabel('Count')
plt.show(block=False)

# Feature selection
features = ['year', 'month', 'day', 'temp_min', 'wind', 'precipitation']
target = 'temp_max'
X = weather_data[features]
y = weather_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions using the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean squared error: {mse:.2f}')
print(f'R2 score: {r2:.2f}')

# Example: predict the maximum temperature for a specific date with given features
example_data = {
    'year': [2023],
    'month': [4],
    'day': [30],
    'temp_min': [7],
    'wind': [2.0],
    'precipitation': [0.5],
}

example_df = pd.DataFrame(example_data)
predicted_temp_max = model.predict(example_df)
print(f'Predicted maximum temperature for {example_data["year"][0]}-{example_data["month"][0]}-{example_data["day"][0]}: {predicted_temp_max[0]:.2f}°C')
#Create a scatter plot of actual vs. predicted maximum temperatures
plt.figure()
sns.scatterplot(x=y_test, y=y_pred)
plt.title('Actual vs. Predicted Maximum Temperatures')
plt.xlabel('Actual Maximum Temperature (°C)')
plt.ylabel('Predicted Maximum Temperature (°C)')
plt.show(block=False)

#Create a residual plot of the linear regression model
plt.figure()
sns.residplot(x=y_test, y=y_pred)
plt.title('Residual Plot of the Linear Regression Model')
plt.xlabel('Actual Maximum Temperature (°C)')
plt.ylabel('Residuals')
plt.show(block=False)

#Keep the plots open
plt.show()
