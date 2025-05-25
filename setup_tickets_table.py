import mysql.connector
from db_config import DB_CONFIG

def setup_tickets_table():
    """
    Script to create the tickets table in the database if it doesn't exist.
    This resolves the error: "Table 'wertigo_db.tickets' doesn't exist"
    """
    try:
        # Connect to the database
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Read SQL from the create_tickets_table.sql file
        with open('create_tickets_table.sql', 'r') as sql_file:
            sql_script = sql_file.read()
        
        # Split the script by semicolons to execute each statement
        statements = sql_script.split(';')
        
        for statement in statements:
            # Skip empty statements
            if statement.strip():
                cursor.execute(statement)
        
        # Commit the changes
        conn.commit()
        
        print("Tickets table created successfully!")
        
        # Close the connection
        cursor.close()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"Error creating tickets table: {e}")
        return False

if __name__ == "__main__":
    setup_tickets_table() 