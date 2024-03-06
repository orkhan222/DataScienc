import pypyodbc 
import pandas as pd

cnxn = pypyodbc.connect("Driver={SQL Server Native Client 11.0};"
                        "Server=164.92.133.203;"
                        "Database=AdventureWorks2014FullForAzure;"
                        "uid=rds103;pwd=Sagol123")
df = pd.read_sql_query('select * from [AdventureWorks2014FullForAzure].[HumanResources].[Shift]', cnxn)

print(df.head(1))