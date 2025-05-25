from flask import Blueprint, jsonify, request
import logging
import requests

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
bp = Blueprint('routing_service', __name__, url_prefix='/api')

@bp.route('/routing', methods=['POST'])
def get_routing():
    """
    API endpoint for calculating routes between multiple points.
    
    Expected JSON input:
    {
        "points": [
            {"lat": 14.5995, "lng": 120.9842, "name": "Starting Point"}, 
            {"lat": 14.6760, "lng": 121.0437, "name": "Destination 1"},
            ...
        ],
        "vehicle": "car" // optional, defaults to "car"
    }
    
    Returns route information and points to draw on the map.
    """
    try:
        data = request.get_json()
        
        if not data or 'points' not in data or len(data['points']) < 2:
            return jsonify({'error': 'At least two points are required for routing'}), 400
        
        points = data['points']
        vehicle = data.get('vehicle', 'car')
        
        # GraphHopper API key - replace with your own if needed
        api_key = '07693a69-9493-445a-a2bf-6d035b9329b6'
        
        # Format the points for the GraphHopper API
        point_params = []
        for point in points:
            if 'lat' not in point or 'lng' not in point:
                return jsonify({'error': 'Each point must have lat and lng coordinates'}), 400
            point_params.append(f"point={point['lat']},{point['lng']}")
        
        # Build the URL for the GraphHopper API
        base_url = 'https://graphhopper.com/api/1/route'
        params = '&'.join(point_params)
        url = f"{base_url}?{params}&vehicle={vehicle}&key={api_key}&type=json&points_encoded=false"
        
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            route_data = response.json()
            
            # Extract relevant information for the frontend
            paths = route_data.get('paths', [])
            if not paths:
                return jsonify({'error': 'No route found'}), 404
            
            # Get the first path (the optimal route)
            path = paths[0]
            
            # Prepare the response
            result = {
                'distance': path.get('distance', 0),  # in meters
                'time': path.get('time', 0),  # in milliseconds
                'points': path.get('points', {}).get('coordinates', []),
                'instructions': path.get('instructions', [])
            }
            
            # Add distance in kilometers and time in minutes
            result['distance_km'] = round(result['distance'] / 1000, 1)
            result['time_min'] = round(result['time'] / (1000 * 60), 0)
            
            return jsonify(result)
        else:
            return jsonify({'error': f'GraphHopper API Error: {response.text}'}), response.status_code
                
    except Exception as e:
        logger.error(f"Error calculating route: {e}")
        return jsonify({'error': f'Error calculating route: {str(e)}'}), 500

@bp.route('/route', methods=['POST'])
def calculate_route():
    """
    API endpoint for calculating routes between points.
    This is designed to work with the tracker.html file.
    
    Expected JSON input:
    {
        "points": [[lon1, lat1], [lon2, lat2], ...],  # GeoJSON format [longitude, latitude]
        "names": ["Place 1", "Place 2", ...]  # Optional place names
    }
    
    Returns route information and points to draw on the map.
    """
    try:
        data = request.get_json()
        
        if not data or 'points' not in data or len(data['points']) < 2:
            return jsonify({'error': 'At least two points are required for routing'}), 400
        
        # Convert from GeoJSON format to the format expected by the get_routing function
        geojson_points = data['points']
        names = data.get('names', [])
        
        points = []
        for i, point in enumerate(geojson_points):
            if len(point) < 2:
                return jsonify({'error': 'Each point must have longitude and latitude coordinates'}), 400
            
            # GeoJSON is [longitude, latitude], but we need [latitude, longitude]
            name = names[i] if i < len(names) else f"Point {i+1}"
            points.append({
                'lat': point[1],  # Latitude is second in GeoJSON
                'lng': point[0],  # Longitude is first in GeoJSON
                'name': name
            })
        
        # Reuse the existing routing logic
        vehicle = data.get('vehicle', 'car')
        
        # GraphHopper API key - replace with your own if needed
        api_key = '07693a69-9493-445a-a2bf-6d035b9329b6'
        
        # Format the points for the GraphHopper API
        point_params = []
        for point in points:
            point_params.append(f"point={point['lat']},{point['lng']}")
        
        # Build the URL for the GraphHopper API
        base_url = 'https://graphhopper.com/api/1/route'
        params = '&'.join(point_params)
        url = f"{base_url}?{params}&vehicle={vehicle}&key={api_key}&type=json&points_encoded=false"
        
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            route_data = response.json()
            
            # Extract relevant information for the frontend
            paths = route_data.get('paths', [])
            if not paths:
                return jsonify({'error': 'No route found'}), 404
            
            # Get the first path (the optimal route)
            path = paths[0]
            
            # Prepare the response
            result = {
                'distance': path.get('distance', 0),  # in meters
                'time': path.get('time', 0),  # in milliseconds
                'points': path.get('points', {}).get('coordinates', []),
                'instructions': path.get('instructions', [])
            }
            
            # Add distance in kilometers and time in minutes
            result['distance_km'] = round(result['distance'] / 1000, 1)
            result['time_min'] = round(result['time'] / (1000 * 60), 0)
            
            # Return the route data in a format the tracker.html file expects
            return jsonify({
                'route': {
                    'distance_km': result['distance_km'],
                    'time_min': result['time_min'],
                    'points': result['points'],
                    'instructions': result['instructions']
                }
            })
        else:
            return jsonify({'error': f'GraphHopper API Error: {response.text}'}), response.status_code
                
    except Exception as e:
        logger.error(f"Error calculating route: {e}")
        return jsonify({'error': f'Error calculating route: {str(e)}'}), 500

@bp.route('/geocode', methods=['GET'])
def geocode():
    """
    API endpoint for geocoding place names to coordinates.
    
    Expected query parameters:
    - q: The query text (e.g., "Intramuros, Manila")
    - limit: (optional) Maximum number of results to return (default: 5)
    
    Returns a list of geocoded results with coordinates.
    """
    try:
        query = request.args.get('q', '')
        limit = request.args.get('limit', 5, type=int)
        
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        # GraphHopper API key
        api_key = '07693a69-9493-445a-a2bf-6d035b9329b6'
        
        # Build the URL for the GraphHopper Geocoding API
        url = f"https://graphhopper.com/api/1/geocode?q={query}&limit={limit}&key={api_key}"
        
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            geocode_data = response.json()
            hits = geocode_data.get('hits', [])
            
            # Format the results
            results = []
            for hit in hits:
                results.append({
                    'name': hit.get('name', ''),
                    'country': hit.get('country', ''),
                    'city': hit.get('city', ''),
                    'state': hit.get('state', ''),
                    'street': hit.get('street', ''),
                    'housenumber': hit.get('housenumber', ''),
                    'point': {
                        'lat': hit.get('point', {}).get('lat', 0),
                        'lng': hit.get('point', {}).get('lng', 0)
                    }
                })
            
            return jsonify({'results': results})
        else:
            return jsonify({'error': f'Geocoding API Error: {response.text}'}), response.status_code
            
    except Exception as e:
        logger.error(f"Error geocoding: {e}")
        return jsonify({'error': f'Error geocoding: {str(e)}'}), 500

@bp.route('/route_for_itinerary', methods=['POST'])
def route_for_itinerary():
    """
    API endpoint for generating routes for a full itinerary day.
    
    Expected JSON input:
    {
        "day": {
            "places": [
                {"name": "Place 1", "city": "City 1", "category": "Category 1"},
                {"name": "Place 2", "city": "City 2", "category": "Category 2"},
                ...
            ]
        },
        "startingPoint": {"lat": 14.5995, "lng": 120.9842, "name": "Hotel"} // optional
    }
    
    Returns complete route information for the day.
    """
    try:
        data = request.get_json()
        
        if not data or 'day' not in data or 'places' not in data['day'] or len(data['day']['places']) < 1:
            return jsonify({'error': 'At least one place is required in the itinerary'}), 400
        
        # Extract places from the itinerary
        places = data['day']['places']
        
        # Starting point (optional)
        starting_point = data.get('startingPoint', None)
        
        # We need to geocode each place to get coordinates
        points = []
        
        # Add starting point if provided
        if starting_point and 'lat' in starting_point and 'lng' in starting_point:
            points.append({
                'lat': starting_point['lat'],
                'lng': starting_point['lng'],
                'name': starting_point.get('name', 'Starting Point')
            })
        
        # Geocode each place in the itinerary
        for place in places:
            # Skip places that already have coordinates
            if ('latitude' in place and place['latitude'] and 
                'longitude' in place and place['longitude']):
                points.append({
                    'lat': place['latitude'],
                    'lng': place['longitude'],
                    'name': place.get('name', 'Destination')
                })
                continue
                
            place_name = place.get('name', '')
            city = place.get('city', '')
            
            if not place_name:
                continue
                
            # Build geocoding query
            query = f"{place_name}, {city}" if city else place_name
            
            try:
                # Call our own geocoding endpoint to avoid code duplication
                geocode_url = f"{request.host_url.rstrip('/')}/api/geocode?q={query}&limit=1"
                geocode_response = requests.get(geocode_url)
                
                if geocode_response.status_code == 200:
                    geocode_data = geocode_response.json()
                    
                    if geocode_data.get('results') and len(geocode_data['results']) > 0:
                        result = geocode_data['results'][0]
                        points.append({
                            'lat': result['point']['lat'],
                            'lng': result['point']['lng'],
                            'name': place_name
                        })
                
            except Exception as e:
                # Log the error but continue with other places
                logger.error(f"Error geocoding {place_name}: {str(e)}")
        
        # If we couldn't geocode any places, return an error
        if len(points) < 2:
            if starting_point and len(points) == 1:
                return jsonify({'error': 'Could not geocode any places in the itinerary'}), 400
            else:
                return jsonify({'error': 'At least two points are required for routing'}), 400
        
        # Call our routing endpoint
        try:
            routing_url = f"{request.host_url.rstrip('/')}/api/routing"
            routing_response = requests.post(
                routing_url, 
                json={'points': points, 'vehicle': 'car'},
                headers={'Content-Type': 'application/json'}
            )
            
            if routing_response.status_code == 200:
                route_data = routing_response.json()
                
                # Enhance the response with place information
                result = {
                    'route': route_data,
                    'points': points,
                    'total_distance_km': route_data.get('distance_km', 0),
                    'total_time_min': route_data.get('time_min', 0)
                }
                
                return jsonify(result)
            else:
                return jsonify({'error': f'Routing error: {routing_response.text}'}), routing_response.status_code
                
        except Exception as e:
            logger.error(f"Error generating route for itinerary: {e}")
            return jsonify({'error': f'Error generating route for itinerary: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error processing itinerary route request: {e}")
        return jsonify({'error': f'Error processing itinerary route request: {str(e)}'}), 500 