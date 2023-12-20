import pandas as pd
from datetime import datetime


def extract_csv(filename,section_name="World news",start=None,end=None):
    
    df = pd.read_csv(filename)
    df = df[df["sectionName"] == section_name]
    
    df = df[df['bodyContent'].notna()]
    
    df["webPublicationDate"] = pd.to_datetime(df["webPublicationDate"])
    
    if start:
        start = pd.to_datetime(start).tz_localize('UTC')
        df = df[df["webPublicationDate"] >= start]
    if end:
        end = pd.to_datetime(end).tz_localize('UTC')
        df = df[df["webPublicationDate"] <= end]
    
    return list(df["webTitle"]), list(df["bodyContent"]), df["webPublicationDate"]


