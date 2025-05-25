import mysql.connector
from mysql.connector import pooling
import json
from db_config import DB_CONFIG
from datetime import datetime
import uuid
import time

# Create a connection pool for better performance
try:
    cnx_pool = mysql.connector.pooling.MySQLConnectionPool(
        pool_name="wertigo_pool",
        pool_size=5,
        **DB_CONFIG
    )
    print("Database connection pool created successfully.")
except Exception as e:
    print(f"Error creating connection pool: {e}")
    # Fallback to direct connections if pooling fails
    cnx_pool = None

def get_connection():
    """Get a connection from the pool or create a new one"""
    try:
        if cnx_pool:
            return cnx_pool.get_connection()
        else:
            return mysql.connector.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Error getting database connection: {e}")
        raise

# User Management Functions
def create_user(username, email, password):
    """Create a new user in the database"""
    try:
        cnx = get_connection()
        cursor = cnx.cursor()
        
        query = ("INSERT INTO users (username, email, password) "
                 "VALUES (%s, %s, %s)")
        cursor.execute(query, (username, email, password))
        
        user_id = cursor.lastrowid
        cnx.commit()
        
        cursor.close()
        cnx.close()
        
        return {"user_id": user_id, "username": username, "email": email}
    except mysql.connector.Error as err:
        if err.errno == 1062:  # Duplicate entry error
            return {"error": "Email already exists"}
        else:
            return {"error": f"Database error: {err}"}

def authenticate_user(email, password):
    """Authenticate a user by email and password"""
    try:
        cnx = get_connection()
        cursor = cnx.cursor(dictionary=True)
        
        query = ("SELECT id, username, email FROM users "
                 "WHERE email = %s AND password = %s")
        cursor.execute(query, (email, password))
        
        user = cursor.fetchone()
        
        cursor.close()
        cnx.close()
        
        if user:
            return {"success": True, "user": user}
        else:
            return {"success": False, "error": "Invalid email or password"}
    except Exception as e:
        return {"success": False, "error": f"Database error: {e}"}

# Session Management Functions
def create_session(user_id=None):
    """Create a new session in the database"""
    try:
        session_id = str(uuid.uuid4())
        
        cnx = get_connection()
        cursor = cnx.cursor()
        
        query = ("INSERT INTO sessions (id, user_id) VALUES (%s, %s)")
        cursor.execute(query, (session_id, user_id))
        
        cnx.commit()
        
        cursor.close()
        cnx.close()
        
        return {"session_id": session_id}
    except Exception as e:
        return {"error": f"Database error: {e}"}

def get_session(session_id):
    """Get session information"""
    try:
        cnx = get_connection()
        cursor = cnx.cursor(dictionary=True)
        
        query = ("SELECT id, user_id, created_at, last_activity FROM sessions "
                 "WHERE id = %s")
        cursor.execute(query, (session_id,))
        
        session = cursor.fetchone()
        
        cursor.close()
        cnx.close()
        
        return session
    except Exception as e:
        print(f"Error getting session: {e}")
        return None

def update_session_activity(session_id):
    """Update the last_activity timestamp of a session"""
    try:
        cnx = get_connection()
        cursor = cnx.cursor()
        
        query = ("UPDATE sessions SET last_activity = CURRENT_TIMESTAMP "
                 "WHERE id = %s")
        cursor.execute(query, (session_id,))
        
        cnx.commit()
        
        cursor.close()
        cnx.close()
        
        return True
    except Exception as e:
        print(f"Error updating session activity: {e}")
        return False

def link_session_to_user(session_id, user_id):
    """Link an existing session to a user after login"""
    try:
        cnx = get_connection()
        cursor = cnx.cursor()
        
        query = ("UPDATE sessions SET user_id = %s "
                 "WHERE id = %s")
        cursor.execute(query, (user_id, session_id))
        
        cnx.commit()
        
        cursor.close()
        cnx.close()
        
        return True
    except Exception as e:
        print(f"Error linking session to user: {e}")
        return False

# Destination Functions
def get_destinations(limit=None, city=None, category=None):
    """
    Get destinations from the database with optional filters
    """
    try:
        # Get a connection from the pool
        cnx = get_connection()
        cursor = cnx.cursor(dictionary=True)
        
        # Build the query
        query = "SELECT * FROM destinations"
        params = []
        
        # Add filters
        conditions = []
        if city:
            conditions.append("city = %s")
            params.append(city)
        if category:
            conditions.append("category = %s")
            params.append(category)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # Add limit
        if limit:
            query += " LIMIT %s"
            params.append(limit)
        
        # Execute query
        cursor.execute(query, tuple(params))
        destinations = cursor.fetchall()
        
        # Close cursor and connection
        cursor.close()
        cnx.close()
        
        return destinations
    except Exception as e:
        print(f"Error getting destinations: {e}")
        return []

def search_destinations(query, limit=10):
    """Search destinations using full-text search"""
    try:
        cnx = get_connection()
        cursor = cnx.cursor(dictionary=True)
        
        search_query = (
            "SELECT id, name, city, province, description, category, "
            "metadata, ratings, budget, latitude, longitude, "
            "operating_hours, contact_information, "
            "MATCH(name, city, description, category, metadata) "
            "AGAINST(%s IN NATURAL LANGUAGE MODE) AS relevance "
            "FROM destinations "
            "WHERE MATCH(name, city, description, category, metadata) "
            "AGAINST(%s IN NATURAL LANGUAGE MODE) "
            "ORDER BY relevance DESC "
            "LIMIT %s"
        )
        
        cursor.execute(search_query, (query, query, limit))
        
        destinations = cursor.fetchall()
        
        cursor.close()
        cnx.close()
        
        return destinations
    except Exception as e:
        print(f"Error searching destinations: {e}")
        return []

def get_distinct_cities():
    """Get a list of all distinct cities in the database"""
    try:
        cnx = get_connection()
        cursor = cnx.cursor()
        
        query = "SELECT DISTINCT city FROM destinations ORDER BY city"
        cursor.execute(query)
        
        cities = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        cnx.close()
        
        return cities
    except Exception as e:
        print(f"Error getting distinct cities: {e}")
        return []

def get_distinct_categories():
    """Get a list of all distinct categories in the database"""
    try:
        cnx = get_connection()
        cursor = cnx.cursor()
        
        query = "SELECT DISTINCT category FROM destinations ORDER BY category"
        cursor.execute(query)
        
        categories = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        cnx.close()
        
        return categories
    except Exception as e:
        print(f"Error getting distinct categories: {e}")
        return []

def get_destination_by_id(destination_id):
    """Get a destination by its ID"""
    try:
        cnx = get_connection()
        cursor = cnx.cursor(dictionary=True)
        
        query = (
            "SELECT id, name, city, province, description, category, "
            "metadata, ratings, budget, latitude, longitude, "
            "operating_hours, contact_information "
            "FROM destinations WHERE id = %s"
        )
        
        cursor.execute(query, (destination_id,))
        
        destination = cursor.fetchone()
        
        cursor.close()
        cnx.close()
        
        return destination
    except Exception as e:
        print(f"Error getting destination by ID: {e}")
        return None

# Trip Functions
def create_trip(user_id, trip_data):
    """Create a new trip in the database"""
    try:
        cnx = get_connection()
        cursor = cnx.cursor()
        
        # Extract trip data
        destination = trip_data.get('destination', '')
        travel_dates = trip_data.get('travel_dates', '')
        travelers = trip_data.get('travelers', 1)
        budget = trip_data.get('budget', '')
        interests = json.dumps(trip_data.get('interests', []))
        
        # Insert trip
        query = (
            "INSERT INTO trips (user_id, destination, travel_dates, travelers, budget, interests) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        )
        
        cursor.execute(query, (user_id, destination, travel_dates, travelers, budget, interests))
        
        trip_id = cursor.lastrowid
        
        # Insert itinerary if provided
        if 'itinerary' in trip_data and trip_data['itinerary']:
            for day_data in trip_data['itinerary']:
                day = day_data.get('day', 1)
                
                # Convert day data to JSON
                itinerary_json = json.dumps(day_data)
                
                # Insert day itinerary
                day_query = (
                    "INSERT INTO trip_itineraries (trip_id, day, itinerary_data) "
                    "VALUES (%s, %s, %s)"
                )
                
                cursor.execute(day_query, (trip_id, day, itinerary_json))
        
        cnx.commit()
        
        cursor.close()
        cnx.close()
        
        return {"trip_id": trip_id}
    except Exception as e:
        print(f"Error creating trip: {e}")
        return {"error": str(e)}

def get_user_trips(user_id):
    """Get all trips for a user"""
    try:
        cnx = get_connection()
        cursor = cnx.cursor(dictionary=True)
        
        # Get basic trip info
        query = (
            "SELECT id, destination, travel_dates, travelers, budget, interests, created_at "
            "FROM trips WHERE user_id = %s "
            "ORDER BY created_at DESC"
        )
        
        cursor.execute(query, (user_id,))
        
        trips = cursor.fetchall()
        
        # For each trip, get its itinerary
        for trip in trips:
            trip_id = trip['id']
            
            itinerary_query = (
                "SELECT day, itinerary_data "
                "FROM trip_itineraries "
                "WHERE trip_id = %s "
                "ORDER BY day"
            )
            
            cursor.execute(itinerary_query, (trip_id,))
            
            itinerary_days = cursor.fetchall()
            
            trip['itinerary'] = []
            for day in itinerary_days:
                # Parse JSON data
                day_data = json.loads(day['itinerary_data'])
                trip['itinerary'].append(day_data)
        
        cursor.close()
        cnx.close()
        
        return trips
    except Exception as e:
        print(f"Error getting user trips: {e}")
        return []

def get_trip_by_id(trip_id, user_id=None):
    """Get a trip by its ID, optionally checking if it belongs to a specific user"""
    try:
        cnx = get_connection()
        cursor = cnx.cursor(dictionary=True)
        
        # Base query
        query = (
            "SELECT id, user_id, destination, travel_dates, travelers, budget, interests, created_at "
            "FROM trips WHERE id = %s"
        )
        
        # Add user check if needed
        params = [trip_id]
        if user_id is not None:
            query += " AND user_id = %s"
            params.append(user_id)
        
        cursor.execute(query, params)
        
        trip = cursor.fetchone()
        
        if trip:
            # Get itinerary
            itinerary_query = (
                "SELECT day, itinerary_data "
                "FROM trip_itineraries "
                "WHERE trip_id = %s "
                "ORDER BY day"
            )
            
            cursor.execute(itinerary_query, (trip_id,))
            
            itinerary_days = cursor.fetchall()
            
            trip['itinerary'] = []
            for day in itinerary_days:
                # Parse JSON data
                day_data = json.loads(day['itinerary_data'])
                trip['itinerary'].append(day_data)
        
        cursor.close()
        cnx.close()
        
        return trip
    except Exception as e:
        print(f"Error getting trip by ID: {e}")
        return None

# Conversation History Functions
def save_conversation(session_id, user_message, system_response, user_id=None):
    """Save a conversation exchange to the database"""
    try:
        cnx = get_connection()
        cursor = cnx.cursor()
        
        query = (
            "INSERT INTO conversations (session_id, user_id, user_message, system_response) "
            "VALUES (%s, %s, %s, %s)"
        )
        
        cursor.execute(query, (session_id, user_id, user_message, system_response))
        
        cnx.commit()
        
        cursor.close()
        cnx.close()
        
        return True
    except Exception as e:
        print(f"Error saving conversation: {e}")
        return False

def get_conversation_history(session_id, limit=10):
    """Get the conversation history for a session"""
    try:
        cnx = get_connection()
        cursor = cnx.cursor(dictionary=True)
        
        query = (
            "SELECT user_message, system_response, timestamp "
            "FROM conversations "
            "WHERE session_id = %s "
            "ORDER BY timestamp DESC "
            "LIMIT %s"
        )
        
        cursor.execute(query, (session_id, limit))
        
        history = cursor.fetchall()
        
        cursor.close()
        cnx.close()
        
        # Reverse to get oldest messages first
        return list(reversed(history))
    except Exception as e:
        print(f"Error getting conversation history: {e}")
        return []

# User Preferences Functions
def save_user_preference(user_id, preference_type, preference_value):
    """Save or update a user preference"""
    try:
        cnx = get_connection()
        cursor = cnx.cursor()
        
        # Check if preference already exists
        check_query = (
            "SELECT id, count FROM preferences "
            "WHERE user_id = %s AND preference_type = %s AND preference_value = %s"
        )
        
        cursor.execute(check_query, (user_id, preference_type, preference_value))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing preference count
            update_query = (
                "UPDATE preferences SET count = count + 1 "
                "WHERE id = %s"
            )
            cursor.execute(update_query, (existing[0],))
        else:
            # Insert new preference
            insert_query = (
                "INSERT INTO preferences (user_id, preference_type, preference_value, count) "
                "VALUES (%s, %s, %s, %s)"
            )
            cursor.execute(insert_query, (user_id, preference_type, preference_value, 1))
        
        cnx.commit()
        
        cursor.close()
        cnx.close()
        
        return True
    except Exception as e:
        print(f"Error saving user preference: {e}")
        return False

def get_user_preferences(user_id, preference_type=None, limit=10):
    """Get user preferences, optionally filtered by type"""
    try:
        cnx = get_connection()
        cursor = cnx.cursor(dictionary=True)
        
        query = (
            "SELECT preference_type, preference_value, count, last_updated "
            "FROM preferences "
            "WHERE user_id = %s"
        )
        
        params = [user_id]
        
        if preference_type:
            query += " AND preference_type = %s"
            params.append(preference_type)
        
        query += " ORDER BY count DESC, last_updated DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        
        preferences = cursor.fetchall()
        
        cursor.close()
        cnx.close()
        
        return preferences
    except Exception as e:
        print(f"Error getting user preferences: {e}")
        return []

# Ticket Functions
def create_ticket(ticket_data):
    conn = get_connection()
    if not conn:
        return {"error": "Database connection failed"}
        
    cursor = conn.cursor()
    
    ticket_id = ticket_data['ticket_id']
    email = ticket_data['email']
    trip_id = ticket_data.get('trip_id')
    itinerary = ticket_data.get('itinerary')
    status = ticket_data.get('status', 'active')
    created_at = ticket_data.get('created_at', datetime.now().isoformat())
    updated_at = ticket_data.get('updated_at', datetime.now().isoformat())
    
    # Convert itinerary to JSON if present
    itinerary_json = json.dumps(itinerary) if itinerary else None
    
    query = """
        INSERT INTO tickets 
        (ticket_id, email, trip_id, itinerary, status, created_at, updated_at) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    
    params = (
        ticket_id,
        email,
        trip_id,
        itinerary_json,
        status,
        created_at,
        updated_at
    )
    
    try:
        cursor.execute(query, params)
        conn.commit()
        return {"ticket_id": ticket_id}
    except mysql.connector.Error as e:
        print(f"Error creating ticket: {e}")
        return {"error": "Failed to create ticket"}
    finally:
        cursor.close()
        conn.close()

def get_ticket(ticket_id):
    conn = get_connection()
    if not conn:
        return None
        
    cursor = conn.cursor(dictionary=True)
    
    query = "SELECT * FROM tickets WHERE ticket_id = %s"
    params = (ticket_id,)
    
    try:
        cursor.execute(query, params)
        result = cursor.fetchone()
        
        if result:
            # Parse JSON data if present
            if result.get('itinerary'):
                try:
                    result['itinerary'] = json.loads(result['itinerary'])
                except json.JSONDecodeError:
                    result['itinerary'] = {}
                    
        return result
    except mysql.connector.Error as e:
        print(f"Error getting ticket: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def get_tickets_by_email(email):
    conn = get_connection()
    if not conn:
        return []
        
    cursor = conn.cursor(dictionary=True)
    
    query = "SELECT * FROM tickets WHERE email = %s ORDER BY created_at DESC"
    params = (email,)
    
    try:
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Parse JSON data for each ticket
        for result in results:
            if result.get('itinerary'):
                try:
                    result['itinerary'] = json.loads(result['itinerary'])
                except json.JSONDecodeError:
                    result['itinerary'] = {}
                    
        return results
    except mysql.connector.Error as e:
        print(f"Error getting tickets by email: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def update_ticket_status(ticket_id, status):
    conn = get_connection()
    if not conn:
        return {"error": "Database connection failed"}
        
    cursor = conn.cursor()
    
    updated_at = datetime.now().isoformat()
    
    query = "UPDATE tickets SET status = %s, updated_at = %s WHERE ticket_id = %s"
    params = (status, updated_at, ticket_id)
    
    try:
        cursor.execute(query, params)
        conn.commit()
        
        if cursor.rowcount > 0:
            return {"success": True, "ticket_id": ticket_id, "status": status}
        else:
            return {"error": "Ticket not found or status not updated"}
    except mysql.connector.Error as e:
        print(f"Error updating ticket status: {e}")
        return {"error": "Failed to update ticket status"}
    finally:
        cursor.close()
        conn.close()

def delete_ticket(ticket_id, email=None):
    """
    Delete a ticket from the database
    
    Args:
        ticket_id (str): The ID of the ticket to delete
        email (str, optional): Email for verification. If provided, will only delete if it matches
    
    Returns:
        dict: Result of the operation with success or error message
    """
    conn = get_connection()
    if not conn:
        return {"error": "Database connection failed"}
        
    cursor = conn.cursor()
    
    # First verify the ticket exists and optionally check email
    verify_query = "SELECT id FROM tickets WHERE ticket_id = %s"
    verify_params = [ticket_id]
    
    if email:
        verify_query += " AND email = %s"
        verify_params.append(email)
    
    try:
        cursor.execute(verify_query, verify_params)
        ticket = cursor.fetchone()
        
        if not ticket:
            if email:
                return {"error": "Ticket not found or email doesn't match"}
            else:
                return {"error": "Ticket not found"}
        
        # Ticket exists and email matches (if provided), proceed with deletion
        delete_query = "DELETE FROM tickets WHERE ticket_id = %s"
        cursor.execute(delete_query, (ticket_id,))
        conn.commit()
        
        if cursor.rowcount > 0:
            return {"success": True, "message": "Ticket successfully deleted"}
        else:
            return {"error": "Failed to delete ticket"}
            
    except mysql.connector.Error as e:
        print(f"Error deleting ticket: {e}")
        return {"error": f"Database error: {str(e)}"}
    finally:
        cursor.close()
        conn.close() 