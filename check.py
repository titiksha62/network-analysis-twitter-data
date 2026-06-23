import pandas as pd
df = pd.read_csv('pheme_master.csv')
print("Total rows:", len(df))
print("Value counts for 'thread_type':\n", df['thread_type'].value_counts())
print("Value counts for 'veracity':\n", df['veracity'].value_counts())