import pandas as pd
import json

def df_to_format(df):
    system_prompt = "Given the medical description report, classify it into one of these categories: " + \
                 "[Cardiovascular / Pulmonary, Gastroenterology, Neurology, Radiology, Surgery]"
    formatted_data = []    
    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        entry = {"messages": [{"role": "system", "content": system_prompt},
                              {"role": "user", "content": row["report"]},
                              {"role": "assistant", "content": row["medical_specialty"]}]}
        formatted_data.append(entry)
    return formatted_data

medical_reports = pd.read_csv("reports.csv")
medical_reports.dropna(subset=['report'], inplace=True)
medical_reports.info()
grouped_data = medical_reports.groupby("medical_specialty").sample(110, random_state=42) # Sample 110 items from each class
val_test_data = grouped_data.groupby("medical_specialty").sample(10, random_state=42)  # sample 10 items from the above data
val = val_test_data.groupby("medical_specialty").head(5) # Take the first 5 of each class
test = val_test_data.groupby("medical_specialty").tail(5) # Take the last 5 of each class
train = grouped_data[~grouped_data.index.isin(val_test_data.index)] # Take the remaining ones for training

print(f"Number of unique medical specialties: {train['medical_specialty'].nunique()}")
print("\nDistribution of reports across medical specialties:")
print(train['medical_specialty'].value_counts())

data = df_to_format(train)
with open('fine_tuning_data.jsonl', 'w') as f:
    for entry in data:
        f.write(json.dumps(entry))
        f.write("\n")

val_data = df_to_format(val)
with open('fine_tuning_data_val.jsonl', 'w') as f:
    for entry in val_data:
        f.write(json.dumps(entry))
        f.write("\n")

test_data = df_to_format(test)
with open('fine_tuning_data_test.jsonl', 'w') as f:
    for entry in test_data:
        f.write(json.dumps(entry))
        f.write("\n")