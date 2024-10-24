import pandas as pd
import re
import io
import numpy as np 


#Read and open files 
#Finanicial Fraud Semi-Supervised Dataset Week4
SFFSD_path= "C:/Users/Raed Karkoub/Desktop/DS340/Data/S-FFSD_p3.csv"
#Credit Card Dataset Week5
CCD_path= "C:/Users/Raed Karkoub/Desktop/DS340/Data/creditcard_p3.csv"

def clean_data(file_path: str) -> io.StringIO:
    data = []
    headers = []
    with open(file_path) as fh:
        data = fh.readlines()
        headers = data.pop(0)
        
        #remove empty line
        data = [item for item in data if item != "\n"]
        
        # remove [, ], and ; from data
        data = [re.sub(r'[\[\];]', "", item) for item in data]
        
        # remove leading white space
        data = [item.lstrip() for item in data]
        
        # replace unequal whitespace with ,
        data = [','.join(item.split()) + "\n" for item in data]

    data_str = "".join(data)
    return io.StringIO(data_str)

#create the dataframes
SSFSD_clean = clean_data(SFFSD_path)
SSFSD_df= pd.read_csv(SSFSD_clean, names=["Time", "Source", "Target", "Amount", "Location", "Type", "Labels"])

CCD_clean = clean_data(CCD_path)
CCD_df= pd.read_csv(CCD_clean, names=["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26","V27", "V28", "Amount", "Class"])
print(SSFSD_df.head())
print(CCD_df.head())


