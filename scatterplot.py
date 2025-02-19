import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\kathr\OneDrive\Documents\MSU25\Spring\DataMining\Project1\Raisin_Dataset\Raisin_Dataset.xlsx'
df = pd.read_excel(file_path)

# Select the relevant columns for scatter plots
area = df['Area']
major_axis_length = df['MajorAxisLength']
eccentricity = df['Eccentricity']
perimeter = df['Perimeter']

# Set up a 3x2 grid of plots
plt.figure(figsize=(12, 10))

# Scatter plot for Area vs MajorAxisLength
plt.subplot(2, 2, 1)
plt.scatter(area, major_axis_length, alpha=0.6, edgecolors='w', s=80)
plt.title("Area vs Major Axis Length")
plt.xlabel("Area")
plt.ylabel("Major Axis Length")

# Scatter plot for Area vs Eccentricity
plt.subplot(2, 2, 2)
plt.scatter(area, eccentricity, alpha=0.6, edgecolors='w', s=80)
plt.title("Area vs Eccentricity")
plt.xlabel("Area")
plt.ylabel("Eccentricity")

# Scatter plot for MajorAxisLength vs Perimeter
plt.subplot(2, 2, 3)
plt.scatter(major_axis_length, perimeter, alpha=0.6, edgecolors='w', s=80)
plt.title("Major Axis Length vs Perimeter")
plt.xlabel("Major Axis Length")
plt.ylabel("Perimeter")

# Scatter plot for Eccentricity vs Perimeter
plt.subplot(2, 2, 4)
plt.scatter(eccentricity, perimeter, alpha=0.6, edgecolors='w', s=80)
plt.title("Eccentricity vs Perimeter")
plt.xlabel("Eccentricity")
plt.ylabel("Perimeter")


# Adjust layout and display the plots
plt.show()

