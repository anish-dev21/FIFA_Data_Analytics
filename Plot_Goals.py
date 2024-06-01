import pandas as pd
import matplotlib.pyplot as plt

# Specify the sheet names containing the data
sheet_names = ["Data", "DEF", "MID", "OFF"]

# Create subplots for each scatter plot
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# Iterate over each sheet and create scatter plot
for idx, sheet_name in enumerate(sheet_names):
    # Read the DataFrame for the current sheet
    df = pd.read_excel("Data.xlsx", sheet_name=sheet_name)
    
    # Convert 'Rating' and 'Fifa Ability Overall' columns to numeric, ignoring errors
    df['Goals'] = pd.to_numeric(df['Goals'], errors='coerce')
    df['Fifa Ability Overall'] = pd.to_numeric(df['Fifa Ability Overall'], errors='coerce')
    
    # Drop rows with NaN values in either column
    df = df.dropna(subset=['Goals', 'Fifa Ability Overall'])
    
    # Plot the scatter plot for 'Rating' vs 'Fifa Ability Overall'
    row = idx // 2
    col = idx % 2
    axs[row, col].scatter(df['Fifa Ability Overall'], df['Goals'])
    axs[row, col].set_title(sheet_name)
    axs[row, col].set_ylabel('Goals')
    axs[row, col].set_xlabel('Fifa Ability Overall')

# Adjust layout
plt.tight_layout()
plt.show()