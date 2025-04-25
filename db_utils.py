import pyodbc
import pandas as pd

# Define connection parameters
server = 'daredb.cba'  
database = 'DARE'  
connection_string = f'DRIVER={{ODBC Driver 13 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'

class DbUtil():
    def __init__(self):
        # Establish a connection
        try:
            self.connection = pyodbc.connect(connection_string)
            # print("Connection successful!")
        except Exception as e:
            print(f"Error: {e}")

    def execute(self, query, fetch_size=100):
        # print("***query:", query)
        connection = self.connection
        cursor = connection.cursor()

        # Execute a query
        cursor.execute(query)

        # Fetch all rows
        rows = cursor.fetchmany(fetch_size)

        # Get column names
        column_names = [column[0] for column in cursor.description]

        # Convert rows to a Pandas DataFrame
        df = pd.DataFrame.from_records(rows, columns=column_names)

        # print("***query result:", df)
        return df

    def clean_up(self):
        # Close the connection
        self.connection.close()

def execute_sql(query):
    try:
        db = DbUtil()
        df = db.execute(query)
        db.clean_up()

        return df.to_string(index=False)
    except Exception as e:
        return f"SQL execution error: {e}"
