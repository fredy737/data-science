# Interfaces with SQLite
# Contains functions that can be called to perform SQL queries, run commands, and show tables

import pandas as pd
import sqlite3

# This function takes an SQL query as an argument, returns a pandas Dataframe of that query
# Function ensures we do not accidentally make changes to the database if one of our queries has an error
# Access 'conn' inside an indented block

def run_query(database, q):
    with sqlite3.connect(database) as conn:
        return pd.read_sql(q, conn)
    
# Take SQL command as an argument, execute it using the sqlite module
# Runs commands like "CREATE VIEW"

def run_command(database, c):
    with sqlite3.connect(database) as conn:
        conn.isolation_level = None    # Autocommit any changes
        conn.execute(c)
        
# Call the run_query() function to return a list of all tables and views in the database

def show_tables(database):
    q = '''
    SELECT
        name,
        type
    FROM sqlite_master
    WHERE type IN ("table","view");
    '''
    
    return run_query(database, q)