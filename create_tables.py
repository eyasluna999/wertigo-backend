import mysql.connector
from db_config import DB_CONFIG, test_connection
import os
import numpy as np
import pandas as pd
import sys
import traceback

def setup_database():
    """Set up the database and tables with better error handling"""
    print("Setting up database...")
    
    # First test the connection
    if not test_connection():
        print("Database connection test failed. Exiting.")
        return False
    
    try:
        # Connect to the database
        cnx = mysql.connector.connect(**DB_CONFIG)
        cursor = cnx.cursor()
        
        # Check if destinations table already exists
        cursor.execute("SHOW TABLES LIKE 'destinations'")
        destinations_exists = cursor.fetchone() is not None
        
        if destinations_exists:
            print("Tables already exist. Skipping table creation.")
            print("Loading embeddings info...")
            load_embeddings_info(cnx, cursor)
            
            print("Loading sample data...")
            load_sample_data(cnx, cursor)
            
            cursor.close()
            cnx.close()
            print("Database setup complete!")
            return True
        
        # Create destinations table with correct data types
        print("Creating destinations table...")
        destinations_table = """
        CREATE TABLE IF NOT EXISTS destinations (
                id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            city VARCHAR(255) NOT NULL,
            province VARCHAR(255),
            description TEXT,
            category VARCHAR(100),
            metadata TEXT,
            ratings FLOAT,
            budget VARCHAR(100),
            latitude FLOAT,
            longitude FLOAT,
            operating_hours VARCHAR(255),
            contact_information TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FULLTEXT KEY destination_search (name, city, description, category, metadata)
        ) ENGINE=InnoDB;
            """
        cursor.execute(destinations_table)
        print("Destinations table created or already exists.")
        
        # Create sessions table for tracking user sessions
        print("Creating sessions table...")
        sessions_table = """
        CREATE TABLE IF NOT EXISTS sessions (
            id VARCHAR(36) PRIMARY KEY,
            user_id INT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            data JSON,
            INDEX (user_id)
        ) ENGINE=InnoDB;
        """
        cursor.execute(sessions_table)
        print("Sessions table created or already exists.")
        
        # Create tickets table
        print("Creating tickets table...")
        tickets_table = """
        CREATE TABLE IF NOT EXISTS tickets (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ticket_id VARCHAR(10) UNIQUE NOT NULL,
            email VARCHAR(255) NOT NULL,
            trip_id INT NULL,
            itinerary JSON,
            status ENUM('active', 'confirmed', 'cancelled', 'completed', 'pending') DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX (email),
            INDEX (ticket_id)
        ) ENGINE=InnoDB;
        """
        cursor.execute(tickets_table)
        print("Tickets table created or already exists.")
        
        # Create model_metadata table to track embedding information
        print("Creating model_metadata table...")
        model_metadata_table = """
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            embedding_path VARCHAR(255) NOT NULL,
            dimensions INT NOT NULL,
            num_destinations INT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            active BOOLEAN DEFAULT TRUE,
            UNIQUE KEY unique_model (model_name)
        ) ENGINE=InnoDB;
        """
        cursor.execute(model_metadata_table)
        print("Model metadata table created or already exists.")
        
        # Load embeddings info
        print("Loading embeddings info...")
        load_embeddings_info(cnx, cursor)
        
        # Load sample data
        print("Loading sample data...")
        load_sample_data(cnx, cursor)
        
        # Close connection
        cursor.close()
        cnx.close()
        
        print("Database setup complete!")
        return True
        
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return False

def load_embeddings_info(cnx=None, cursor=None):
    """Load model embedding information into database"""
    close_connection = False
    
    try:
        # Create connection if not provided
        if cnx is None or cursor is None:
            cnx = mysql.connector.connect(**DB_CONFIG)
            cursor = cnx.cursor()
            close_connection = True
        
        # Get embedding information
        dest_embed_path = os.path.join('model_output', 'destination_embeddings.npy')
        
        if os.path.exists(dest_embed_path):
            dest_embeddings = np.load(dest_embed_path)
            print(f"Loaded destination embeddings: {dest_embeddings.shape}")
            
            # Insert/update model metadata using alias to avoid deprecated VALUES function
            query = """
            INSERT INTO model_metadata 
                (model_name, embedding_path, dimensions, num_destinations, last_updated) 
            VALUES 
                (%s, %s, %s, %s, NOW()) AS new_values
            ON DUPLICATE KEY UPDATE
                embedding_path = new_values.embedding_path,
                dimensions = new_values.dimensions,
                num_destinations = new_values.num_destinations,
                last_updated = NOW(),
                active = TRUE
            """
            
            dims = dest_embeddings.shape[1] if len(dest_embeddings.shape) > 1 else 0
            num_dest = dest_embeddings.shape[0] if len(dest_embeddings.shape) > 0 else 0
            
            try:
                cursor.execute(query, ('roberta_destination_model', dest_embed_path, dims, num_dest))
                cnx.commit()
                print(f"Updated model metadata for destination embeddings")
            except mysql.connector.Error as err:
                # Handle MySQL 5.x which doesn't support the alias syntax
                if err.errno == 1064:  # Syntax error
                    print("Using alternative syntax for older MySQL versions")
                    # Try the older syntax
                    query = """
                    REPLACE INTO model_metadata 
                        (model_name, embedding_path, dimensions, num_destinations, last_updated, active) 
                    VALUES 
                        (%s, %s, %s, %s, NOW(), TRUE)
                    """
                    cursor.execute(query, ('roberta_destination_model', dest_embed_path, dims, num_dest))
                    cnx.commit()
                    print(f"Updated model metadata for destination embeddings (using REPLACE)")
                else:
                    raise
            
        else:
            print(f"Warning: Destination embeddings file not found at {dest_embed_path}")
        
    except Exception as e:
        print(f"Error loading embeddings info: {e}")
        traceback.print_exc()
    finally:
        # Close connection if we created it
        if close_connection and cnx is not None and cursor is not None:
            cursor.close()
            cnx.close()

def load_sample_data(cnx=None, cursor=None):
    """Load sample destination data from CSV if available"""
    close_connection = False
    
    try:
        # Create connection if not provided
        if cnx is None or cursor is None:
            cnx = mysql.connector.connect(**DB_CONFIG)
            cursor = cnx.cursor()
            close_connection = True
        
        csv_path = os.path.join('final_dataset.csv')
        if os.path.exists(csv_path):
            print(f"Loading sample data from {csv_path}")
            
            # Read data
            df = pd.read_csv(csv_path)
            
            # Clean data
            df = df.fillna("")
            
            # Check if destinations table is empty
            cursor.execute("SELECT COUNT(*) FROM destinations")
            count = cursor.fetchone()[0]
            
            if count == 0:
                print("Destinations table is empty, inserting sample data...")
                
                # Insert records
                inserted = 0
                errors = 0
                
                for _, row in df.iterrows():
                    try:
                        query = """
                        INSERT INTO destinations 
                            (name, city, province, description, category, metadata, 
                             ratings, budget, latitude, longitude, operating_hours, contact_information)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        
                        # Convert ratings to float
                        try:
                            ratings = float(row['ratings']) if row['ratings'] else None
                        except (ValueError, TypeError):
                            ratings = None
                            
                        # Convert budget to string
                        try:
                            budget = str(row['budget']) if row['budget'] else None
                        except (ValueError, TypeError):
                            budget = None
                            
                        # Convert coordinates to float
                        try:
                            latitude = float(row['latitude']) if row['latitude'] else None
                        except (ValueError, TypeError):
                            latitude = None
                            
                        try:
                            longitude = float(row['longitude']) if row['longitude'] else None
                        except (ValueError, TypeError):
                            longitude = None
                        
                        # Check for column name variations
                        op_hours = row.get('operating hours', row.get('operating_hours', ''))
                        contact_info = row.get('contact information', row.get('contact_information', ''))
                        
                        data = (
                            row['name'],
                            row['city'],
                            row['province'],
                            row['description'],
                            row['category'],
                            row['metadata'],
                            ratings,
                            budget,
                            latitude,
                            longitude,
                            op_hours,
                            contact_info
                        )
                        
                        cursor.execute(query, data)
                        inserted += 1
                        
                        # Commit in smaller batches to prevent timeout
                        if inserted % 20 == 0:
                            cnx.commit()
                            print(f"Inserted {inserted} records so far...")
                            
                    except Exception as e:
                        print(f"Error inserting row: {e}")
                        errors += 1
                
                # Final commit
                cnx.commit()
                print(f"Inserted {inserted} sample destinations, with {errors} errors")
            else:
                print(f"Destinations table already has {count} records, skipping sample data import")
            
        else:
            print(f"Sample data file not found at {csv_path}")
        
    except Exception as e:
        print(f"Error loading sample data: {e}")
        traceback.print_exc()
    finally:
        # Close connection if we created it
        if close_connection and cnx is not None and cursor is not None:
            cursor.close()
            cnx.close()

if __name__ == "__main__":
    setup_database() 