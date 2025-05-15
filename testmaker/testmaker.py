import pandas as pd

df = pd.read_csv("Heart_disease_cleveland_new.csv")
df = df.drop(columns=["target"]) 
df.to_csv("test_verisi.csv", index=False)  