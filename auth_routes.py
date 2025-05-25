from flask import Blueprint, request, jsonify
import database as db
import json
import uuid

# Create a Blueprint for the auth routes
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/api/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        # Check if required fields are provided
        if not all(k in data for k in ['username', 'email', 'password']):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Create the user
        result = db.create_user(data['username'], data['email'], data['password'])
        
        if 'error' in result:
            return jsonify(result), 400
        
        # Create a session for the new user
        session_result = db.create_session(result['user_id'])
        
        if 'error' in session_result:
            return jsonify(session_result), 500
        
        # Combine user and session data
        response = {
            "success": True,
            "user": {
                "id": result['user_id'],
                "username": result['username'],
                "email": result['email']
            },
            "session_id": session_result['session_id']
        }
        
        return jsonify(response), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/api/login', methods=['POST'])
def login():
    """Log in an existing user"""
    try:
        data = request.get_json()
        
        # Check if required fields are provided
        if not all(k in data for k in ['email', 'password']):
            return jsonify({"error": "Missing email or password"}), 400
        
        # Authenticate the user
        auth_result = db.authenticate_user(data['email'], data['password'])
        
        if not auth_result['success']:
            return jsonify(auth_result), 401
        
        # Create a session for the user
        session_result = db.create_session(auth_result['user']['id'])
        
        if 'error' in session_result:
            return jsonify(session_result), 500
        
        # If the user already had a guest session, link it to the user
        if 'previous_session_id' in data and data['previous_session_id']:
            db.link_session_to_user(data['previous_session_id'], auth_result['user']['id'])
            
        # Combine user and session data
        response = {
            "success": True,
            "user": auth_result['user'],
            "session_id": session_result['session_id']
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get session information"""
    try:
        session = db.get_session(session_id)
        
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        # Update the session's last activity timestamp
        db.update_session_activity(session_id)
        
        # If the session is linked to a user, get user info
        user = None
        if session['user_id']:
            # In a real app, you'd get more user info, but for simplicity
            # we'll just return the user ID
            user = {"id": session['user_id']}
        
        response = {
            "session_id": session['id'],
            "created_at": session['created_at'].isoformat(),
            "last_activity": session['last_activity'].isoformat(),
            "user": user
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/api/trips', methods=['GET'])
def get_trips():
    """Get all trips for the authenticated user"""
    try:
        # Get the session ID from the query parameters
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({"error": "Session ID required"}), 400
        
        # Get the session
        session = db.get_session(session_id)
        
        if not session:
            return jsonify({"error": "Invalid session"}), 401
        
        # Make sure the session is linked to a user
        if not session['user_id']:
            return jsonify({"error": "Please log in to view trips"}), 401
        
        # Update the session's last activity timestamp
        db.update_session_activity(session_id)
        
        # Get the user's trips
        trips = db.get_user_trips(session['user_id'])
        
        return jsonify({"trips": trips}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/api/trips/<int:trip_id>', methods=['GET'])
def get_trip(trip_id):
    """Get a specific trip"""
    try:
        # Get the session ID from the query parameters
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({"error": "Session ID required"}), 400
        
        # Get the session
        session = db.get_session(session_id)
        
        if not session:
            return jsonify({"error": "Invalid session"}), 401
        
        # Update the session's last activity timestamp
        db.update_session_activity(session_id)
        
        # If the session is linked to a user, only let them access their own trips
        user_id = session['user_id']
        
        # Get the trip
        trip = db.get_trip_by_id(trip_id, user_id)
        
        if not trip:
            return jsonify({"error": "Trip not found"}), 404
        
        return jsonify({"trip": trip}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/api/trips', methods=['POST'])
def create_trip():
    """Create a new trip"""
    try:
        # Get the session ID from the query parameters or headers
        session_id = request.args.get('session_id') or request.headers.get('X-Session-ID')
        
        if not session_id:
            return jsonify({"error": "Session ID required"}), 400
        
        # Get the trip data from the request body
        trip_data = request.get_json()
        
        if not trip_data:
            return jsonify({"error": "Trip data required"}), 400
        
        # Get the session
        session = db.get_session(session_id)
        
        if not session:
            return jsonify({"error": "Invalid session"}), 401
        
        # Make sure the session is linked to a user
        if not session['user_id']:
            return jsonify({"error": "Please log in to create trips"}), 401
        
        # Update the session's last activity timestamp
        db.update_session_activity(session_id)
        
        # Create the trip
        result = db.create_trip(session['user_id'], trip_data)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500 