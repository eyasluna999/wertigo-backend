from flask import Blueprint, jsonify, request
import database as db
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
bp = Blueprint('trip_service', __name__, url_prefix='/api')

@bp.route('/create_trip', methods=['POST'])
def create_trip():
    """
    Create a trip itinerary based on the given destination and preferences
    """
    try:
        # Get data from request
        data = request.get_json()
        destination = data.get('destination')
        travelers = data.get('travelers', 1)
        budget = data.get('budget', 'mid-range')
        interests = data.get('interests', [])
        
        # Get complete destination data if provided
        selected_destination_data = data.get('selected_destination_data')
        
        # Get all selected destinations if provided
        all_selected_destinations = data.get('all_selected_destinations', [])
        
        # Validate destination
        if not destination:
            return jsonify({"error": "Destination is required"}), 400
        
        # Use all selected destinations if provided
        if all_selected_destinations and len(all_selected_destinations) > 0:
            destinations = all_selected_destinations
            main_destination = destination
        # Check if we have complete destination data
        elif selected_destination_data:
            # Use the provided destination data directly
            main_destination = selected_destination_data.get('city', destination)
            
            # Create a list with the selected destination
            destinations = [selected_destination_data]
            
            # Add more destinations from the same city
            additional_destinations = db.get_destinations(limit=10, city=main_destination)
            
            # Filter out the selected destination if it's in the additional destinations
            if additional_destinations:
                additional_destinations = [dest for dest in additional_destinations 
                                         if dest.get('id') != selected_destination_data.get('id')]
                destinations.extend(additional_destinations)
        else:
            # Check if the destination exists in our database
            # First check for exact match
            destinations = db.get_destinations(limit=10, city=destination)
            
            # If no exact match, try a search
            if not destinations:
                destinations = db.search_destinations(destination, limit=10)
                
            # If still no match, suggest alternatives
            if not destinations:
                # Get all cities as suggestions
                cities = db.get_distinct_cities()
                
                # Return error with suggestions
                return jsonify({
                    "error": "Destination not found",
                    "suggestions": cities[:10]  # Limit to top 10
                }), 404
            
            # Use the first destination city as our main destination
            main_destination = destinations[0]['city']
        
        # Generate itinerary based on number of selected destinations
        # If user has selected destinations, use those to determine days
        if all_selected_destinations and len(all_selected_destinations) > 0:
            # Calculate number of days based on selected destinations
            # Aim for 2-3 destinations per day
            num_destinations = len(all_selected_destinations)
            destinations_per_day = 2  # Target 2 destinations per day
            num_days = max(1, (num_destinations + destinations_per_day - 1) // destinations_per_day)
            
            # Generate itinerary using user-selected destinations
            itinerary = generate_itinerary(all_selected_destinations, num_days)
        else:
            # Default to 3 days if no specific destinations were selected
            num_days = 3
            itinerary = generate_itinerary(destinations, num_days)
        
        # Get activities based on interests
        if interests:
            # Convert interests list to search string
            interest_query = " ".join(interests)
            activity_recommendations = db.search_destinations(interest_query, limit=20)
            
            # Only include activities in the same city
            activities = [act for act in activity_recommendations 
                         if act['city'].lower() == main_destination.lower()]
            
            # Add these to each day's itinerary if they're not already included
            existing_place_ids = []
            for day in itinerary:
                existing_place_ids.extend([place['id'] for place in day['places']])
            
            # Distribute remaining activities across days
            for i, activity in enumerate(activities):
                if activity['id'] not in existing_place_ids:
                    day_index = i % len(itinerary)
                    # Add only if we have space (maximum 4 activities per day)
                    if len(itinerary[day_index]['places']) < 4:
                        itinerary[day_index]['places'].append({
                            'id': activity['id'],
                            'name': activity['name'],
                            'category': activity['category'],
                            'description': activity['description'],
                            'rating': activity['ratings'] if activity['ratings'] is not None else 4.0,
                            'latitude': activity.get('latitude'),
                            'longitude': activity.get('longitude'),
                            'operating_hours': activity.get('operating_hours'),
                            'contact_information': activity.get('contact_information'),
                            'budget': activity.get('budget', 'mid-range')
                        })
        
        # Create trip response with all destination data
        trip = {
            "destination": main_destination,
            "travelers": travelers,
            "budget": budget,
            "interests": interests,
            "itinerary": itinerary,
            "selected_destinations": all_selected_destinations
        }
        
        # Save to database if user is authenticated
        session_id = request.args.get('session_id') or request.headers.get('X-Session-ID')
        if session_id:
            session = db.get_session(session_id)
            if session and session['user_id']:
                # Save trip to database
                db.create_trip(session['user_id'], trip)
                
        return jsonify({"trip": trip}), 200
        
    except Exception as e:
        logger.error(f"Error creating trip: {e}")
        return jsonify({"error": str(e)}), 500

# Helper function to generate a simple itinerary based on recommendations
def generate_itinerary(recommendations, num_days=3):
    if not recommendations:
        return []
    
    # Create a simple day-by-day itinerary
    itinerary = []
    
    # Calculate how many places to include per day
    total_places = len(recommendations)
    places_per_day = max(1, min(4, (total_places + num_days - 1) // num_days))
    
    for day in range(1, num_days + 1):
        # Calculate start and end indices for this day's places
        start_idx = (day - 1) * places_per_day
        end_idx = min(start_idx + places_per_day, total_places)
        
        # Select places for this day
        day_places = recommendations[start_idx:end_idx]
        
        # Skip if no places for this day
        if not day_places:
            continue
            
        daily_plan = {
            'day': day,
            'places': [],
            'meals': [
                {'type': 'Breakfast', 'suggestion': 'Local breakfast options'},
                {'type': 'Lunch', 'suggestion': 'Restaurant near ' + day_places[0]['name']},
                {'type': 'Dinner', 'suggestion': 'Evening dining experience'}
            ],
            'transportation': 'Local transport or taxi',
            'estimated_cost': calculate_estimate_cost(day_places)
        }
        
        # Add all relevant place data with full metadata
        for place in day_places:
            place_data = {
                'id': place.get('id'),
                'name': place.get('name'),
                'city': place.get('city'),
                'category': place.get('category'),
                'description': place.get('description'),
                'rating': place.get('ratings') if place.get('ratings') is not None else 4.0,
                'budget': place.get('budget', 'mid-range')
            }
            
            # Add optional metadata if available
            if 'latitude' in place and place['latitude']:
                place_data['latitude'] = place['latitude']
            if 'longitude' in place and place['longitude']:
                place_data['longitude'] = place['longitude']
            if 'operating_hours' in place and place['operating_hours']:
                place_data['operating_hours'] = place['operating_hours']
            if 'contact_information' in place and place['contact_information']:
                place_data['contact_information'] = place['contact_information']
            if 'metadata' in place and place['metadata']:
                place_data['metadata'] = place['metadata']
            if 'province' in place and place['province']:
                place_data['province'] = place['province']
            
            # Add to daily places
            daily_plan['places'].append(place_data)
        
        itinerary.append(daily_plan)
    
    return itinerary

# Helper function to calculate estimated cost
def calculate_estimate_cost(places):
    # Very simple cost estimator based on number of places and type
    base_cost = 1000  # Base cost in PHP
    for place in places:
        if 'category' in place:
            category = place['category'].lower()
            if 'resort' in category:
                base_cost += 3000
            elif 'historical' in category:
                base_cost += 500
            elif 'museum' in category:
                base_cost += 800
            elif 'natural' in category:
                base_cost += 1200
            else:
                base_cost += 1000
        else:
            base_cost += 1000
    
    return f"₱{base_cost}"

@bp.route('/trips', methods=['GET'])
def get_user_trips():
    """Get all trips for a user"""
    try:
        # Get user ID from session
        session_id = request.args.get('session_id') or request.headers.get('X-Session-ID')
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
            
        session = db.get_session(session_id)
        if not session or not session['user_id']:
            return jsonify({"error": "Invalid session or user not authenticated"}), 401
            
        user_id = session['user_id']
        
        # Get trips from database
        trips = db.get_trips_by_user(user_id)
        
        return jsonify({"trips": trips}), 200
        
    except Exception as e:
        logger.error(f"Error getting user trips: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route('/trips/<trip_id>', methods=['GET'])
def get_trip(trip_id):
    """Get a specific trip by ID"""
    try:
        # Get trip from database
        trip = db.get_trip_by_id(trip_id)
        
        if not trip:
            return jsonify({"error": "Trip not found"}), 404
            
        return jsonify({"trip": trip}), 200
        
    except Exception as e:
        logger.error(f"Error getting trip: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route('/trips/<trip_id>', methods=['PUT'])
def update_trip(trip_id):
    """Update a trip"""
    try:
        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Get user ID from session
        session_id = request.headers.get('X-Session-ID')
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
            
        session = db.get_session(session_id)
        if not session or not session['user_id']:
            return jsonify({"error": "Invalid session or user not authenticated"}), 401
            
        user_id = session['user_id']
        
        # Verify ownership of trip
        trip = db.get_trip_by_id(trip_id)
        if not trip or trip['user_id'] != user_id:
            return jsonify({"error": "Trip not found or you don't have permission to update it"}), 403
            
        # Update trip
        result = db.update_trip(trip_id, data)
        
        if 'error' in result:
            return jsonify(result), 400
            
        return jsonify({"message": "Trip updated successfully", "trip": result}), 200
        
    except Exception as e:
        logger.error(f"Error updating trip: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route('/trips/<trip_id>', methods=['DELETE'])
def delete_trip(trip_id):
    """Delete a trip"""
    try:
        # Get user ID from session
        session_id = request.headers.get('X-Session-ID')
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
            
        session = db.get_session(session_id)
        if not session or not session['user_id']:
            return jsonify({"error": "Invalid session or user not authenticated"}), 401
            
        user_id = session['user_id']
        
        # Verify ownership of trip
        trip = db.get_trip_by_id(trip_id)
        if not trip or trip['user_id'] != user_id:
            return jsonify({"error": "Trip not found or you don't have permission to delete it"}), 403
            
        # Delete trip
        result = db.delete_trip(trip_id)
        
        if 'error' in result:
            return jsonify(result), 400
            
        return jsonify({"message": "Trip deleted successfully"}), 200
        
    except Exception as e:
        logger.error(f"Error deleting trip: {e}")
        return jsonify({"error": str(e)}), 500 