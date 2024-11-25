import json
import pandas as pd

# Path to the JSON file
file_path = "un_dataset.json"

# Load the JSON data
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# List to store the English terms
english_terms = []

# Extract "label_en" and "alt_labels_en" from each node
for node in data.get("nodes", []):
    # Get "label_en" if it exists
    if "label_en" in node:
        english_terms.append(node["label_en"])
    # Get all "alt_labels_en" if they exist
    if "alt_labels_en" in node and isinstance(node["alt_labels_en"], list):
        english_terms.extend(node["alt_labels_en"])

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(english_terms, columns=["English Terms"])
df.to_csv("english_terms.csv", index=False)

# Print a message
print("English terms have been extracted and saved to 'english_terms.csv'.")