from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

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

# Create FastAPI app
app = FastAPI(
    title="WerTigo Travel Recommendation API",
    description="API for travel recommendations and trip planning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:3000',
        'http://localhost:8000', 
        'http://127.0.0.1:3000',
        'http://127.0.0.1:8000',
        'http://localhost:5500',
        'http://127.0.0.1:5500',
        'https://*.vercel.app',
        'https://*.railway.app',
        'https://*.huggingface.co'
    ],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Register blueprints only if available
if auth_available:
    app.include_router(auth_bp)
if recommender_available:
    app.include_router(recommender.bp)
if trip_service_available:
    app.include_router(trip_service.bp)
if routing_service_available:
    app.include_router(routing_service.bp)
if ticket_service_available:
    app.include_router(ticket_service.bp)

# Initialize tokenizer
try:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
except Exception as e:
    logger.error(f"Error loading tokenizer: {e}")
    tokenizer = None

# Session management
active_sessions = {}

# Pydantic models for request/response
class SessionRequest(BaseModel):
    user_id: Optional[str] = None

class RecommendationRequest(BaseModel):
    session_id: str
    query: str
    city: Optional[str] = None
    category: Optional[str] = None

# Root route
@app.get("/")
async def root():
    """Simple root endpoint to show API is running"""
    return {
        'message': 'WerTigo Travel Recommendation API',
        'status': 'running',
        'version': '1.0',
        'environment': 'production',
        'endpoints': [
            '/api/health',
            '/api/create_session',
            '/api/recommend',
            '/api/dataset/info'
        ]
    }

@app.post("/api/create_session")
async def create_session(request: SessionRequest):
    """Create a new session for a user"""
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {
        'created_at': datetime.now(),
        'last_activity': datetime.now(),
        'user_id': request.user_id
    }
    return {'session_id': session_id}

@app.get("/api/session/{session_id}")
async def validate_session(session_id: str):
    """Validate a session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    if datetime.now() - session['last_activity'] > timedelta(hours=1):
        del active_sessions[session_id]
        raise HTTPException(status_code=401, detail="Session expired")
    
    session['last_activity'] = datetime.now()
    return {'valid': True, 'session': session}

@app.post("/api/recommend")
async def get_recommendations(request: RecommendationRequest):
    """Get travel recommendations"""
    # Validate session
    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[request.session_id]
    if datetime.now() - session['last_activity'] > timedelta(hours=1):
        del active_sessions[request.session_id]
        raise HTTPException(status_code=401, detail="Session expired")
    
    session['last_activity'] = datetime.now()
    
    # Check city and category availability
    availability = check_city_category_availability(request.city, request.category, df)
    if not availability['exists']:
        return availability
    
    # Get recommendations
    try:
        # Check if model is available
        if model is None or df is None or embeddings is None:
            return JSONResponse(
                content={
                    'is_conversation': True,
                    'message': "I'm sorry, but the recommendation system is currently unavailable. The model needs to be trained first. Please run 'python revised.py' to train the model, then restart the server."
                },
                status_code=503
            )
        
        if tokenizer is None:
            return JSONResponse(
                content={
                    'is_conversation': True,
                    'message': "I'm sorry, but there's an issue with the text processing system. Please check the server logs."
                },
                status_code=503
            )
            
        # Get available cities and categories from the dataframe
        available_cities = df['city'].unique().tolist()
        available_categories = df['category'].unique().tolist()
        
        # Extract query information
        try:
            from revised import extract_query_info
            city, category, budget, clean_query, sentiment_info, budget_amount = extract_query_info(
                request.query,
                available_cities,
                available_categories
            )
        except Exception as e:
            logger.error(f"Error extracting query info: {e}")
            return JSONResponse(
                content={
                    'is_conversation': True,
                    'message': "I had trouble understanding your query. Could you please rephrase it?"
                },
                status_code=400
            )
        
        # Get recommendations
        try:
            from revised import get_recommendations
            recommendations, scores = get_recommendations(
                clean_query if clean_query else request.query,
                tokenizer,
                model,
                embeddings,
                df,
                city=city,
                category=category,
                budget=budget,
                budget_amount=budget_amount,
                top_n=5
            )
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return JSONResponse(
                content={
                    'is_conversation': True,
                    'message': "I encountered an error while searching for recommendations. Please try again with a different query."
                },
                status_code=500
            )
        
        if recommendations.empty:
            return JSONResponse(
                content={
                    'is_conversation': True,
                    'message': "I couldn't find any places matching your criteria. Could you try being more specific or adjusting your filters?"
                },
                status_code=200
            )
            
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
                'detected_budget_type': 'strict' if any(word in request.query.lower() for word in ['under', 'below', 'less than']) else 'flexible'
            },
            'data_availability': {
                'city_exists': True,
                'category_exists': True,
                'combination_exists': True,
                'total_results': len(formatted_recs)
            }
        }
        
        return JSONResponse(
            content=response,
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error processing recommendation request: {e}")
        return JSONResponse(
            content={
                'is_conversation': True,
                'message': "I'm having trouble processing your request right now. Please try again in a moment."
            },
            status_code=500
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'data_loaded': df is not None,
        'embeddings_loaded': embeddings is not None
    }

@app.get("/api/dataset/info")
async def get_dataset_info():
    """Get information about the dataset"""
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    return {
        'total_destinations': len(df),
        'available_cities': df['city'].unique().tolist(),
        'available_categories': df['category'].unique().tolist()
    }

def check_city_category_availability(city, category, dataframe):
    """Check if a specific city-category combination exists in the dataset"""
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
    
    return {
        'exists': True,
        'message': "City and category combination is valid",
        'available_categories': available_categories,
        'available_cities': available_cities
    }

if __name__ == '__main__':
    # Log startup information
    logger.info("Starting WerTigo Travel API - Local Development...")
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Data loaded: {df is not None}")
    logger.info(f"Embeddings loaded: {embeddings is not None}")
    logger.info(f"Tokenizer loaded: {tokenizer is not None}")
    
    if model is None:
        logger.warning("Model is not loaded. Please run 'python revised.py' to train the model first.")
    
    # Run FastAPI app locally
    logger.info("Server running at: http://localhost:5000")
    import uvicorn
    uvicorn.run(app, host='localhost', port=5000) 