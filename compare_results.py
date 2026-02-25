import pandas as pd

#load the csv files
phising_df=pd.read_csv('uploads/phising.csv')
prediction_df=pd.read_csv('uploads/prediction.csv')

if 'Result' in phising_df.columns and 'Result' not  in prediction_df.columns:
    raise ValueError("Both files should have the 'Result' column for comparison.")

phising_df['Result'] = phising_df['Result'].replace(-1,0)
prediction_df['Result'] = prediction_df['Result'].replace(-1,0)

matching_results = phising_df['Result'] == prediction_df['Result']

total_records = len(phising_df)

print(f"Total Records: {total_records}")
print(f"Matching Records: {matching_results.sum()}")

accuracy = (matching_results.sum() / total_records) * 100
print(f"Accuracy: {accuracy:.2f}%")