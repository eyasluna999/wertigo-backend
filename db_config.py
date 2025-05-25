# MySQL Database Configuration
import os

# Database connection settings
DB_CONFIG = {
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', '1234'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'wertigo_db'),
    'port': int(os.getenv('DB_PORT', '3306')),
    'raise_on_warnings': True,
    'auth_plugin': 'mysql_native_password'  # Explicitly specify authentication plugin
}

# Function to test database connection
def test_connection():
    """Test the database connection with the specified config"""
    import mysql.connector
    
    try:
        print("Attempting to connect to MySQL database...")
        # Create a connection without specifying the database first
        config_without_db = DB_CONFIG.copy()
        config_without_db.pop('database', None)
        
        cnx = mysql.connector.connect(**config_without_db)
        print("Successfully connected to MySQL!")
        
        # Check if database exists
        cursor = cnx.cursor()
        cursor.execute(f"SHOW DATABASES LIKE '{DB_CONFIG['database']}'")
        result = cursor.fetchone()
        
        if result:
            print(f"Database '{DB_CONFIG['database']}' exists!")
            
            # Try connecting to the specific database
            cnx.close()
            cnx = mysql.connector.connect(**DB_CONFIG)
            print(f"Successfully connected to database '{DB_CONFIG['database']}'!")
            cnx.close()
            
            return True
        else:
            print(f"Database '{DB_CONFIG['database']}' does not exist. Creating it...")
            cursor.execute(f"CREATE DATABASE {DB_CONFIG['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print(f"Database '{DB_CONFIG['database']}' created!")
            cnx.close()
            
            # Try connecting to the new database
            cnx = mysql.connector.connect(**DB_CONFIG)
            print(f"Successfully connected to new database '{DB_CONFIG['database']}'!")
            cnx.close()
            
            return True
            
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        
        if err.errno == 1045:  # Access denied error
            print("Authentication failed. Check your username and password in db_config.py")
        elif err.errno == 2003:  # Can't connect to MySQL server
            print("Cannot connect to MySQL server. Make sure the server is running and accessible.")
        
        return False
        
if __name__ == "__main__":
    # If this file is run directly, test the connection
    test_connection() 