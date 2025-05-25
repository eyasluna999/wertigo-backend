from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import numpy as np
import torch
from transformers import RobertaTokenizer
from model_handler import model, df, embeddings
import uuid
from datetime import datetime, timedelta
import random
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import application components
from model_handler import init_model
try:
    from knowledge_cache import knowledge_cache
except ImportError:
    logger.warning("knowledge_cache module not found")
    knowledge_cache = None

try:
    from auth_routes import auth_bp
    auth_available = True
except ImportError:
    logger.warning("auth_routes module not found")
    auth_available = False

try:
    import recommender
    recommender_available = True
except ImportError:
    logger.warning("recommender module not found")
    recommender_available = False

try:
    import trip_service
    trip_service_available = True
except ImportError:
    logger.warning("trip_service module not found")
    trip_service_available = False

try:
    import routing_service
    routing_service_available = True
except ImportError:
    logger.warning("routing_service module not found")
    routing_service_available = False

try:
    import ticket_service
    ticket_service_available = True
except ImportError:
    logger.warning("ticket_service module not found")
    ticket_service_available = False

# Create Flask app
app = Flask(__name__)

# Configure CORS for local development and production
CORS(app, 
     origins=[
         'http://localhost:3000',
         'http://localhost:8000', 
         'http://127.0.0.1:3000',
         'http://127.0.0.1:8000',
         'http://localhost:5500',  # VS Code Live Server
         'http://127.0.0.1:5500',   # VS Code Live Server
         'https://*.vercel.app',    # Vercel deployments
         'https://*.railway.app'    # Railway deployments
     ],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization', 'Cache-Control', 'X-Request-ID'],
     supports_credentials=True)

# Register blueprints only if available
if auth_available:
    app.register_blueprint(auth_bp)
if recommender_available:
    app.register_blueprint(recommender.bp)
if trip_service_available:
    app.register_blueprint(trip_service.bp)
if routing_service_available:
    app.register_blueprint(routing_service.bp)
if ticket_service_available:
    app.register_blueprint(ticket_service.bp)

# Initialize tokenizer
try:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
except Exception as e:
    logger.error(f"Error loading tokenizer: {e}")
    tokenizer = None

# Session management
active_sessions = {}

# Root route
@app.route('/', methods=['GET'])
def root():
    """Simple root endpoint to show API is running"""
    return jsonify({
        'message': 'WerTigo Travel Recommendation API - Local Development',
        'status': 'running',
        'version': '1.0',
        'environment': 'local',
        'endpoints': [
            '/api/health',
            '/api/create_session',
            '/api/recommend',
            '/api/dataset/info'
        ]
    })

def check_city_category_availability(city, category, dataframe):
    """
    Check if a specific city-category combination exists in the dataset
    Returns: dict with 'exists', 'message', 'available_categories', 'available_cities'
    """
    if dataframe is None or dataframe.empty:
        return {
            'exists': False,
            'message': "Dataset is not available at the moment.",
            'available_categories': [],
            'available_cities': []
        }
    
    # Get all available cities and categories
    available_cities = dataframe['city'].unique().tolist()
    available_categories = dataframe['category'].unique().tolist()
    
    # If both city and category are specified
    if city and category:
        # Check if city exists in dataset
        city_exists = any(c.lower() == city.lower() for c in available_cities)
        
        if not city_exists:
            return {
                'exists': False,
                'message': f"Sorry, I don't have data for {city}. Available cities include: {', '.join(available_cities[:5])}{'...' if len(available_cities) > 5 else ''}",
                'available_categories': available_categories,
                'available_cities': available_cities
            }
        
        # Check if category exists in dataset
        category_exists = any(c.lower() == category.lower() for c in available_categories)
        
        if not category_exists:
            return {
                'exists': False,
                'message': f"Sorry, I don't have data for {category} category. Available categories include: {', '.join(available_categories[:5])}{'...' if len(available_categories) > 5 else ''}",
                'available_categories': available_categories,
                'available_cities': available_cities
            }
        
        # Check if the specific city-category combination exists
        city_category_mask = (
            (dataframe['city'].str.lower() == city.lower()) & 
            (dataframe['category'].str.lower() == category.lower())
        )
        
        if not city_category_mask.any():
            # Get available categories for this city
            city_mask = dataframe['city'].str.lower() == city.lower()
            available_categories_in_city = dataframe[city_mask]['category'].unique().tolist()
            
            return {
                'exists': False,
                'message': f"{city} has no {category} categories in our dataset. Available categories in {city}: {', '.join(available_categories_in_city)}",
                'available_categories': available_categories_in_city,
                'available_cities': available_cities
            }
    
    # If only city is specified
    elif city:
        city_exists = any(c.lower() == city.lower() for c in available_cities)
        
        if not city_exists:
            return {
                'exists': False,
                'message': f"Sorry, I don't have data for {city}. Available cities include: {', '.join(available_cities[:5])}{'...' if len(available_cities) > 5 else ''}",
                'available_categories': available_categories,
                'available_cities': available_cities
            }
        
        # Get available categories for this city
        city_mask = dataframe['city'].str.lower() == city.lower()
        available_categories_in_city = dataframe[city_mask]['category'].unique().tolist()
        
        return {
            'exists': True,
            'message': f"Found data for {city}. Available categories: {', '.join(available_categories_in_city)}",
            'available_categories': available_categories_in_city,
            'available_cities': available_cities
        }
    
    # If only category is specified
    elif category:
        category_exists = any(c.lower() == category.lower() for c in available_categories)
        
        if not category_exists:
            return {
                'exists': False,
                'message': f"Sorry, I don't have data for {category} category. Available categories include: {', '.join(available_categories[:5])}{'...' if len(available_categories) > 5 else ''}",
                'available_categories': available_categories,
                'available_cities': available_cities
            }
        
        # Get available cities for this category
        category_mask = dataframe['category'].str.lower() == category.lower()
        available_cities_in_category = dataframe[category_mask]['city'].unique().tolist()
        
        return {
            'exists': True,
            'message': f"Found {category} places in: {', '.join(available_cities_in_category)}",
            'available_categories': available_categories,
            'available_cities': available_cities_in_category
        }
    
    # If neither city nor category is specified
    return {
        'exists': True,
        'message': "Please specify a city or category for recommendations.",
        'available_categories': available_categories,
        'available_cities': available_cities
    }

@app.route('/api/create_session', methods=['POST'])
def create_session():
    try:
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {
            'created_at': datetime.now(),
            'last_active': datetime.now(),
            'expires_at': datetime.now() + timedelta(hours=24)
        }
        logger.info(f"Created new session: {session_id}")
        return jsonify({'session_id': session_id})
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({'error': 'Failed to create session'}), 500

@app.route('/api/session/<session_id>', methods=['GET'])
def validate_session(session_id):
    try:
        if session_id not in active_sessions:
            return jsonify({'error': 'Session not found'}), 404
            
        session = active_sessions[session_id]
        
        # Check if session has expired
        if datetime.now() > session['expires_at']:
            del active_sessions[session_id]
            return jsonify({'error': 'Session expired'}), 401
            
        # Update last active timestamp
        session['last_active'] = datetime.now()
        
        return jsonify({
            'session_id': session_id,
            'created_at': session['created_at'].isoformat(),
            'last_active': session['last_active'].isoformat(),
            'expires_at': session['expires_at'].isoformat()
        })
    except Exception as e:
        logger.error(f"Error validating session: {e}")
        return jsonify({'error': 'Failed to validate session'}), 500

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    try:
        # Check if model is available
        if model is None or df is None or embeddings is None:
            return jsonify({
                'is_conversation': True,
                'message': "I'm sorry, but the recommendation system is currently unavailable. The model needs to be trained first. Please run 'python revised.py' to train the model, then restart the server."
            }), 503
        
        if tokenizer is None:
            return jsonify({
                'is_conversation': True,
                'message': "I'm sorry, but there's an issue with the text processing system. Please check the server logs."
            }), 503
            
        data = request.get_json()
        query = data.get('query', '')
        session_id = data.get('session_id')
        limit = data.get('limit', 5)
        
        # Validate session if provided
        if session_id and session_id not in active_sessions:
            return jsonify({'error': 'Invalid session'}), 401
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        # Get available cities and categories from the dataframe
        available_cities = df['city'].unique().tolist()
        available_categories = df['category'].unique().tolist()
        
        # Extract query information
        try:
            from revised import extract_query_info
            city, category, budget, clean_query, sentiment_info, budget_amount = extract_query_info(
                query,
                available_cities,
                available_categories
            )
        except Exception as e:
            logger.error(f"Error extracting query info: {e}")
            return jsonify({
                'is_conversation': True,
                'message': "I had trouble understanding your query. Could you please rephrase it?"
            }), 400
        
        # Check if the city-category combination exists in the dataset
        availability_check = check_city_category_availability(city, category, df)
        
        if not availability_check['exists']:
            # Return a specific message about data availability
            return jsonify({
                'is_conversation': True,
                'message': availability_check['message'],
                'detected_city': city,
                'detected_category': category,
                'available_cities': availability_check['available_cities'],
                'available_categories': availability_check['available_categories'],
                'data_availability': {
                    'city_exists': city in availability_check['available_cities'] if city else None,
                    'category_exists': category in availability_check['available_categories'] if category else None,
                    'combination_exists': False
                }
            })
        
        # Get recommendations
        try:
            from revised import get_recommendations
            recommendations, scores = get_recommendations(
                clean_query if clean_query else query,
                tokenizer,
                model,
                embeddings,
                df,
                city=city,
                category=category,
                budget=budget,
                budget_amount=budget_amount,
                top_n=limit
            )
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return jsonify({
                'is_conversation': True,
                'message': "I encountered an error while searching for recommendations. Please try again with a different query."
            }), 500
        
        if recommendations.empty:
            return jsonify({
                'is_conversation': True,
                'message': "I couldn't find any places matching your criteria. Could you try being more specific or adjusting your filters?"
            })
            
        # Format recommendations
        formatted_recs = []
        for i, (idx, row) in enumerate(recommendations.iterrows()):
            rec = {
                'id': idx,
                'name': row['name'],
                'city': row['city'],
                'province': row.get('province', ''),
                'category': row['category'],
                'description': row['description'],
                'rating': float(row.get('ratings', 0)) if row.get('ratings') else 0,
                'budget': row.get('budget', 'Not specified'),
                'operating_hours': row.get('operating_hours', 'Not specified'),
                'contact_information': row.get('contact_information', 'Not specified'),
                'latitude': float(row.get('latitude', 0)) if row.get('latitude') else 0,
                'longitude': float(row.get('longitude', 0)) if row.get('longitude') else 0,
                'similarity_score': float(scores[i]),
                'image_path': f"img/location/{row['name']}/{random.randint(1, 3)}.jpg"
            }
            formatted_recs.append(rec)
            
        # Add detected information
        response = {
            'recommendations': formatted_recs,
            'detected_city': city,
            'detected_category': category,
            'detected_budget': budget_amount if budget_amount else budget,
            'query_understanding': {
                'detected_category_type': category.lower() if category else None,
                'detected_budget_type': 'strict' if any(word in query.lower() for word in ['under', 'below', 'less than']) else 'flexible'
            },
            'data_availability': {
                'city_exists': True,
                'category_exists': True,
                'combination_exists': True,
                'total_results': len(formatted_recs)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing recommendation request: {e}")
        return jsonify({
            'is_conversation': True,
            'message': "I'm having trouble processing your request right now. Please try again in a moment."
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify system status"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_loaded': df is not None and not df.empty,
        'embeddings_loaded': embeddings is not None,
        'tokenizer_loaded': tokenizer is not None,
    }
    
    # Check components properly for DataFrames
    components_loaded = [
        model is not None,
        df is not None and not df.empty,
        embeddings is not None,
        tokenizer is not None
    ]
    
    if not all(components_loaded):
        status['status'] = 'degraded'
        status['message'] = 'Some components are not available. Model may need training.'
    
    return jsonify(status)

@app.route('/api/dataset/info', methods=['GET'])
def get_dataset_info():
    """Get information about available cities and categories in the dataset"""
    try:
        if df is None or df.empty:
            return jsonify({
                'error': 'Dataset not available'
            }), 503
        
        # Get unique cities and categories
        cities = df['city'].unique().tolist()
        categories = df['category'].unique().tolist()
        
        # Get city-category combinations
        city_category_combinations = []
        for city in cities:
            city_data = df[df['city'] == city]
            city_categories = city_data['category'].unique().tolist()
            city_category_combinations.append({
                'city': city,
                'categories': city_categories,
                'count': len(city_data)
            })
        
        return jsonify({
            'total_destinations': len(df),
            'total_cities': len(cities),
            'total_categories': len(categories),
            'cities': cities,
            'categories': categories,
            'city_category_combinations': city_category_combinations
        })
        
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Log startup information
    logger.info("Starting WerTigo Travel API - Local Development...")
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Data loaded: {df is not None}")
    logger.info(f"Embeddings loaded: {embeddings is not None}")
    logger.info(f"Tokenizer loaded: {tokenizer is not None}")
    
    if model is None:
        logger.warning("Model is not loaded. Please run 'python revised.py' to train the model first.")
    
    # Run Flask app locally
    logger.info("Server running at: http://localhost:5000")
    app.run(debug=True, host='localhost', port=5000) 