import pandas as pd

# Load the Excel file
df_off = pd.read_excel('/Users/gab/Desktop/DA_FIFA/FIFA_Data_Analytics/Data.xlsx', sheet_name='OFF')
df_def = pd.read_excel('/Users/gab/Desktop/DA_FIFA/FIFA_Data_Analytics/Data.xlsx', sheet_name='DEF')
df_mid = pd.read_excel('/Users/gab/Desktop/DA_FIFA/FIFA_Data_Analytics/Data.xlsx', sheet_name='MID')

# drop the string columns and ignore the errors
def clean_data(df):
    df = df.drop(columns=['Name', 'Position'])
    df = df.apply(pd.to_numeric, errors ='coerce').dropna()
    return df

# df_off = clean_data(df_off)
# df_mid = clean_data(df_mid)
# df_def = clean_data(df_def)