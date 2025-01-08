import json
import pandas as pd

ID = "id"
ENGLISH = "en"
ARABIC = "ar"
SPANISH = "es"
FRENCH = "fr"
RUSSIAN = "ru"
CHINESE = "zh"

# Path to the JSON file
file_path = "un_dataset.json"

# Load the JSON data
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# List to store the terms for each language
terms = {
    ID: [],
    ENGLISH: [],
    ARABIC: [],
    SPANISH: [],
    FRENCH: [],
    RUSSIAN: [],
    CHINESE: []
}

for node in data.get("nodes", []):
    if len(node["key"]) >= 6:
        terms[ID].append(node["key"])
        terms[ENGLISH].append(node.get("label_en", ""))
        terms[ARABIC].append(node.get("label_ar", ""))
        terms[SPANISH].append(node.get("label_es", ""))
        terms[FRENCH].append(node.get("label_fr", ""))
        terms[RUSSIAN].append(node.get("label_ru", ""))
        terms[CHINESE].append(node.get("label_zh", ""))

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(terms)
df.to_csv("multilingual_terms.csv", index=False)

# Print a message
print("Terms for all languages have been extracted and saved to 'multilingual_terms.csv'.")