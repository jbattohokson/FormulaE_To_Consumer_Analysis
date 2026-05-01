import os
import pandas as pd

# This gets the directory where your script is located
base_path = os.path.dirname(__file__)

# Example: If you upload a CSV named 'data.csv' later
# data_path = os.path.join(base_path, 'data.csv')
# df = pd.read_csv(data_path)

print("Environment setup complete. Ready to analyze Formula E data.")
