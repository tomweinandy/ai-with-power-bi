"""
Scripts used for Chapter 11: Advanced Power BI and AI with Data Science Languages
"""

#################################################################################
# Ingesting Data with Python
#################################################################################

# Import packages
import os
import pandas as pd

# Set directory
filepath = "C:/Users/tomwe/Documents/GoogleTrendsData/"
data_folder = os.path.expanduser(filepath)

# Read the list of files in the specified folder
files = os.listdir(data_folder)

# Extract "multiTimeline" files
google_trends_files = [file for file in files if file[0:13] == "multiTimeline"]

# Identify the most recent date
most_recent_file = max(google_trends_files)

# Load data
df = pd.read_csv(filepath + most_recent_file, skiprows=2)
print(df)


#################################################################################
# Transforming Data with Python
#################################################################################

# Import packages
import pandas as pd

# Add six-month moving average
df_rolling = dataset.set_index("Month")
df_rolling = df_rolling.rolling(6, min_periods=1).mean()
df_rolling = df_rolling.reset_index()

# Rename columns
df_rolling = df_rolling.rename(columns={"Month":"month_year", "python machine learning: (United States)":"python_moving_average", "r machine learning: (United States)": "r_moving_average"})

# Insert original columns to the new dataframe
df_rolling.insert(1, "python_machine_learning", dataset["python machine learning: (United States)"])
df_rolling.insert(2, "r_machine_learning", dataset["r machine learning: (United States)"])

# Encode date to proper format and insert into dataframe
date_column = pd.to_datetime(df_rolling["month_year"]).dt.date
df_rolling.insert(1, "date", date_column)


#################################################################################
# Visualizing Data with Python
#################################################################################

# Import packages
import matplotlib.pyplot as plt

# Format plot
fig = plt.figure(figsize=(16,8), dpi=200)
plt.title("Change in Programming Language Popularity Over Time", fontsize=20)
plt.ylabel("Google Search Frequency Index", fontsize=16)
plt.yticks(fontsize=16)

# Plot moving averages as a line
plt.plot(dataset["month_year"], dataset["python_moving_average"], c="gold", linewidth=3)
plt.plot(dataset["month_year"], dataset["r_moving_average"], c="red", linewidth=3)

# Plot monthly observations as a scatter plot
plt.scatter(x=dataset["month_year"], y=dataset["python_machine_learning"], c="gold", alpha=0.5)
plt.scatter(x=dataset["month_year"], y=dataset["r_machine_learning"], c="red", alpha=0.5)

# Set the x-axis tick positions, labels, and angle
xticks = dataset["month_year"][6:][::12]   # skip first 6 labels (to start in January) then select every 12th label thereafter
xtick_labels = [str(i) for i in xticks]
plt.xticks(xticks, xtick_labels, fontsize=20)
plt.xticks(rotation=30)

# Add the legend
plt.legend(["Python machine learning", "R machine learning"], fontsize=16, frameon=False)

# Display plot
plt.show()


#################################################################################
# Using a Pre-Trained Model with Python on Transform
#################################################################################

# Load packages
import pandas as pd
import pickle
import requests
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Locate pre-trained model from GitHub
algo_url = "https://raw.githubusercontent.com/tomweinandy/AIinPowerBI/main/vending_model.pickle"

# Clean data
df = dataset
df = df.dropna()
df = df.drop(columns=["WeekStartingSat"])

# Load the pickled deep learning algorithm
response = requests.get(algo_url)
model_info = pickle.loads(response.content)

# Reconstruct the model architecture
loaded_model = tf.keras.models.model_from_json(model_info["architecture"])

# Set the model weights
loaded_model.set_weights(model_info["weights"])

# Preprocess the data
label_encoder = LabelEncoder()
df["Location"] = label_encoder.fit_transform(df["Location"])  # Encode location labels

# Split the data into input features (X)
X = df.drop("Location", axis=1)

# Predict the labels
predictions = loaded_model.predict(X)

# Change predictions to integers between 0 and 4
encoded_labels = [int(min(max(p.round(0), 0), 4)) for p in predictions]

# Decode the predicted labels
decoded_labels = label_encoder.inverse_transform(encoded_labels)

# Add the predicted labels to the DataFrame
df.insert(0, "Predicted_Location", decoded_labels)

# Decode Location
df["Location"] = label_encoder.inverse_transform(df["Location"])
