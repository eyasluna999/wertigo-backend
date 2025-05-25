from flask import Blueprint, jsonify, request
from knowledge_cache import knowledge_cache
from model_handler import model, df, embeddings, tokenizer, device
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
import database as db

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
bp = Blueprint('recommender', __name__, url_prefix='/api')

# Import from revised.py for query understanding
try:
    import importlib.util
    import os
    
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    revised_path = os.path.join(current_dir, "revised.py")
    
    spec = importlib.util.spec_from_file_location("travel_model", revised_path)
    travel_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(travel_model)
    extract_query_info = travel_model.extract_query_info
except Exception as e:
    logger.error(f"Error importing extract_query_info from revised.py: {e}")
    # Try fallback to try.py
    try:
        logger.info("Falling back to try.py for extract_query_info...")
        try_path = os.path.join(current_dir, "try.py")
        spec = importlib.util.spec_from_file_location("travel_model", try_path)
        travel_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(travel_model)
        extract_query_info = travel_model.extract_query_info
    except Exception as e2:
        logger.error(f"Error importing from try.py: {e2}")
        def extract_query_info(query, cities, categories, budgets):
            """Fallback function if import fails"""
            return None, None, None, query, None, None

# Enhanced query understanding with context
def understand_query(query_text, session_id=None):
    """
    Enhanced query understanding with improved context and semantic analysis
    """
    query_lower = query_text.lower().strip()
    
    # Enhanced travel intents with cafe-specific patterns
    intents = {
        'find_destination': [
            'show me', 'recommend', 'suggest', 'find', 'looking for', 'want to visit', 
            'places to visit', 'want to see', 'where can i find', 'where is', 'where are',
            'best place for', 'top places in', 'popular spots in', 'coffee places',
            'cafe spots', 'cafes in', 'coffee shops'
        ],
        'explore_activity': [
            'activities', 'things to do', 'what can i do', 'what to do', 'what are the best',
            'where can i', 'how to experience', 'best way to', 'top activities in',
            'best coffee in', 'where to drink coffee', 'cafe hopping'
        ],
        'plan_trip': [
            'plan', 'itinerary', 'schedule', 'trip to', 'travel to', 'want to go',
            'how to plan', 'create itinerary', 'make schedule', 'organize trip'
        ],
        'get_info': [
            'tell me about', 'information about', 'what is', 'details about', 'facts about',
            'describe', 'explain', 'more about', 'background of'
        ],
        'compare_destinations': [
            'compare', 'difference between', 'which is better', 'versus', 'vs',
            'should i visit', 'better option between'
        ]
    }
    
    # Enhanced intent detection with scoring
    detected_intent = None
    intent_score = 0
    
    for intent, phrases in intents.items():
        for phrase in phrases:
            if phrase in query_lower:
                # Improved scoring based on phrase length and position
                current_score = len(phrase.split()) * 2
                if query_lower.startswith(phrase):
                    current_score += 3  # Bonus for phrases at start
                if current_score > intent_score:
                    detected_intent = intent
                    intent_score = current_score
    
    # Enhanced budget indicators with more specific terms
    budget_indicators = {
        'low': [
            'cheap', 'budget', 'affordable', 'inexpensive', 'low cost', 'economical',
            'budget-friendly', 'cost-effective', 'pocket-friendly', 'thrifty'
        ],
        'medium': [
            'moderate', 'reasonable', 'mid-range', 'standard', 'average',
            'balanced', 'modest', 'fair price', 'middle range'
        ],
        'high': [
            'luxury', 'expensive', 'high-end', 'premium', 'exclusive',
            'upscale', 'deluxe', 'first-class', 'top-tier'
        ]
    }
    
    # Enhanced trip type detection
    trip_type_indicators = {
        'adventure': [
            'adventure', 'hiking', 'trekking', 'outdoor', 'extreme', 'activities',
            'thrilling', 'exciting', 'challenging', 'exploration'
        ],
        'relaxation': [
            'relax', 'peaceful', 'quiet', 'calm', 'unwind', 'spa', 'retreat',
            'tranquil', 'serene', 'leisure', 'restful'
        ],
        'cultural': [
            'culture', 'history', 'historical', 'museum', 'tradition', 'heritage',
            'artistic', 'architectural', 'religious', 'spiritual'
        ],
        'family': [
            'family', 'kids', 'children', 'family-friendly', 'child-friendly',
            'suitable for children', 'family activities', 'family vacation'
        ],
        'romantic': [
            'romantic', 'couple', 'honeymoon', 'anniversary', 'date',
            'romantic getaway', 'couple activities', 'romantic spots'
        ],
        'food': [
            'food', 'restaurant', 'cuisine', 'dining', 'eat',
            'local food', 'traditional dishes', 'culinary'
        ],
        'shopping': [
            'shopping', 'market', 'mall', 'store', 'retail',
            'souvenirs', 'local products', 'shopping district'
        ]
    }
    
    # Get user context and preferences with enhanced error handling
    user_context = {}
    preferred_categories = []
    preferred_cities = []
    
    if session_id:
        try:
            user_context = knowledge_cache.get_session_context(session_id) or {}
            preferred_categories = knowledge_cache.get_preferred_categories(session_id) or []
            preferred_cities = knowledge_cache.get_preferred_cities(session_id) or []
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
    
    # Extract city and category with improved error handling
    available_cities = df['city'].unique().tolist() if df is not None else []
    available_categories = df['category'].unique().tolist() if df is not None else []
    
    try:
        city, category, budget, cleaned_query, sentiment_info, budget_amount = extract_query_info(
            query_text, 
            available_cities, 
            available_categories,
            None
        )
        
        # Enhanced context-based fallbacks
        if not city and preferred_cities and 'city' not in query_lower:
            city = preferred_cities[0]
            logger.info(f"Using preferred city from context: {city}")
        
        if not category and preferred_categories and 'category' not in query_lower:
            category = preferred_categories[0]
            logger.info(f"Using preferred category from context: {category}")
        
        if not cleaned_query.strip():
            cleaned_query = query_text
            
    except Exception as e:
        logger.error(f"Error in extract_query_info: {e}")
        city = None
        category = None
        budget = None
        cleaned_query = query_text
        sentiment_info = None
        budget_amount = None
    
    # Enhanced query understanding with more context
    query_understanding = {
        'original_query': query_text,
        'cleaned_query': cleaned_query,
        'detected_intent': detected_intent,
        'city': city,
        'category': category,
        'budget_preference': budget,
        'budget_amount': budget_amount,
        'trip_type': None,  # Will be set below
        'sentiment_info': sentiment_info,
        'user_context': user_context,
        'preferred_categories': preferred_categories,
        'preferred_cities': preferred_cities
    }
    
    # Detect trip type with improved accuracy
    for type_name, terms in trip_type_indicators.items():
        if any(term in query_lower for term in terms):
            query_understanding['trip_type'] = type_name
            break
    
    # Detect budget preference with improved accuracy
    for budget, terms in budget_indicators.items():
        if any(term in query_lower for term in terms):
            query_understanding['budget_preference'] = budget
            break
    
    return query_understanding

# Helper function to handle conversation queries
def handle_conversation(query_text, session_id=None):
    """
    Handle conversational queries that don't require recommendation models
    Returns a response if the query is conversational, None otherwise
    """
    query_lower = query_text.lower().strip()
    
    # Check for greetings
    greetings = ['hi', 'hello', 'hey', 'hi!', 'hello!', 'hey!', 'greetings', 'good morning', 'good afternoon', 'good evening']
    help_phrases = ['help', 'how does this work', 'what can you do', 'what do you do']
    thanks_phrases = ['thanks', 'thank you', 'thx', 'appreciate it']
    
    # Get conversation context safely
    conversation_context = {}
    if session_id:
        try:
            conversation_context = knowledge_cache.get_conversation_context(session_id, query_text)
        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
    
    # Check if this is a greeting or basic conversation
    if any(greeting == query_lower for greeting in greetings):
        # Personalize response based on conversation history
        message = "Hello! I'm your travel assistant. I can recommend destinations based on your preferences."
        
        # If returning user, make it more personalized
        if session_id and 'conversation' in conversation_context and conversation_context['conversation'].get('exchanges', 0) > 3:
            # Reference previous interests
            if 'preferences' in conversation_context and 'categories' in conversation_context['preferences']:
                preferred_categories = conversation_context['preferences']['categories']
                if preferred_categories:
                    message += f" I see you've been interested in {preferred_categories[0]}. Would you like more recommendations in that category?"
            elif 'preferences' in conversation_context and 'cities' in conversation_context['preferences']:
                preferred_cities = conversation_context['preferences']['cities']
                if preferred_cities:
                    message += f" You seem interested in {preferred_cities[0]}. Would you like to explore more about it?"
            else:
                message += " Welcome back! How can I help with your travel plans today?"
        else:
            # First-time or infrequent user
            message += " Try asking about resorts, historical sites, nature spots, or specific cities you're interested in!"
        
        response = {
            'is_conversation': True,
            'message': message
        }
        
        # Add to cache and conversation history
        try:
            knowledge_cache.add(query_text, response, session_id)
            if session_id:
                knowledge_cache.add_to_conversation(session_id, query_text, message)
        except Exception as e:
            logger.error(f"Error updating knowledge cache: {e}")
        
        return response
    
    elif any(phrase in query_lower for phrase in help_phrases):
        # Simple help message as fallback in case of errors
        message = "I can help you find interesting places to visit based on your preferences. You can ask me things like 'Show me beach resorts' or 'I want to visit historical sites'."
        
        # Try to provide a more personalized help message if context is available
        try:
            # Consider conversation history for help message
            if 'preferences' in conversation_context:
                if 'categories' in conversation_context['preferences'] and conversation_context['preferences']['categories']:
                    categories = conversation_context['preferences']['categories']
                    message += f" I notice you like {', '.join(categories[:2])}. You can ask about these specifically."
                elif 'cities' in conversation_context['preferences'] and conversation_context['preferences']['cities']:
                    cities = conversation_context['preferences']['cities']
                    message += f" You seem interested in {', '.join(cities[:2])}. You can ask about destinations there."
            
            message += " You can also rate places (1-5 stars) to get more tailored recommendations."
        except Exception as e:
            logger.error(f"Error generating personalized help message: {e}")
        
        response = {
            'is_conversation': True,
            'message': message
        }
        
        # Add to cache and conversation history
        try:
            knowledge_cache.add(query_text, response, session_id)
            if session_id:
                knowledge_cache.add_to_conversation(session_id, query_text, message)
        except Exception as e:
            logger.error(f"Error updating knowledge cache: {e}")
        
        return response
    
    elif any(phrase in query_lower for phrase in thanks_phrases):
        # Simple thanks response
        message = "You're welcome! Is there anything else you'd like to know about destinations or travel planning?"
        
        response = {
            'is_conversation': True,
            'message': message
        }
        
        # Add to cache and conversation history
        try:
            knowledge_cache.add(query_text, response, session_id)
            if session_id:
                knowledge_cache.add_to_conversation(session_id, query_text, message)
        except Exception as e:
            logger.error(f"Error updating knowledge cache: {e}")
        
        return response
    
    # Check for common conversation patterns with regex
    conversation_patterns = [
        (r'\b(hi|hello|hey|greetings)\b', "Hi there! I'm your travel assistant. How can I help you plan your trip today?"),
        (r'\bhow are you\b', "I'm doing great, thanks for asking! I'm ready to help you discover amazing travel destinations. What kind of place are you looking for?"),
        (r'\bthank you\b', "You're welcome! Is there anything else you'd like to know about travel destinations?"),
        (r'\bgoodbye|bye\b', "Goodbye! Feel free to come back when you're planning your next adventure.")
    ]
    
    for pattern, response in conversation_patterns:
        if re.search(pattern, query_text, re.IGNORECASE):
            return {
                "is_conversation": True,
                "message": response
            }
    
    # Not a conversational query
    return None

# Helper function to update session context with recommendations
def update_session_context_from_recommendations(session_id, recommendations, query):
    """Update session context with information from recommendations"""
    if not session_id or not recommendations:
        return
    
    try:
        context_update = {
            'last_query': query,
            'last_response_timestamp': time.time()
        }
        
        # Extract cities and categories from recommendations
        cities = []
        categories = []
        
        for rec in recommendations[:3]:  # Use top 3 recommendations
            if 'city' in rec and rec['city']:
                cities.append(rec['city'])
            if 'category' in rec and rec['category']:
                categories.append(rec['category'])
        
        # Update context with unique values
        if cities:
            context_update['last_cities'] = list(dict.fromkeys(cities))
        if categories:
            context_update['last_categories'] = list(dict.fromkeys(categories))
        
        knowledge_cache.update_session_context(session_id, context_update)
    except Exception as e:
        logger.error(f"Error updating session context from recommendations: {e}")

# Function to generate follow-up suggestions
def generate_follow_up_suggestions(recommendations, conversation_context):
    """Generate contextual follow-up questions based on recommendations and conversation history"""
    suggestions = []
    
    if not recommendations:
        return suggestions
    
    # Extract key information from the recommendations
    cities = list(set(rec['city'] for rec in recommendations if 'city' in rec))
    categories = list(set(rec['category'] for rec in recommendations if 'category' in rec))
    
    # Generate city-based questions
    if cities:
        primary_city = cities[0]
        suggestions.append(f"What are some popular activities in {primary_city}?")
        
        # If there's conversation history about transportation
        if conversation_context and 'conversation' in conversation_context and 'recent_topics' in conversation_context['conversation']:
            if 'transportation' in conversation_context['conversation']['recent_topics']:
                suggestions.append(f"How can I get around in {primary_city}?")
                
        # If there's conversation history about food
        if conversation_context and 'conversation' in conversation_context and 'recent_topics' in conversation_context['conversation']:
            if 'food' in conversation_context['conversation']['recent_topics']:
                suggestions.append(f"What local food should I try in {primary_city}?")
    
    # Generate category-based questions
    if categories:
        primary_category = categories[0]
        # If accommodation was mentioned
        if conversation_context and 'conversation' in conversation_context and 'recent_topics' in conversation_context['conversation']:
            if 'accommodation' in conversation_context['conversation']['recent_topics']:
                suggestions.append(f"What are the best hotels near these {primary_category}?")
        
        # If budget was mentioned
        if conversation_context and 'conversation' in conversation_context and 'recent_topics' in conversation_context['conversation']:
            if 'budget' in conversation_context['conversation']['recent_topics']:
                suggestions.append(f"What's the typical cost to visit these {primary_category}?")
    
    # If planning was a topic, suggest itinerary creation
    if conversation_context and 'conversation' in conversation_context and 'recent_topics' in conversation_context['conversation']:
        if 'planning' in conversation_context['conversation']['recent_topics'] and cities:
            suggestions.append(f"Can you create an itinerary for {cities[0]}?")
    
    # Take only a few suggestions
    return suggestions[:3]

# Function to find the best category match for a query
def find_best_category_match(query_text, available_categories):
    """
    Find the best matching category for a query using multiple matching strategies
    """
    query_lower = query_text.lower().strip()
    
    # Complete dictionary of category synonyms and variations
    category_synonyms = {
        'Restaurant': ['restaurant', 'dining', 'eatery', 'food', 'cuisine', 'meal', 'bistro', 'diner', 'eating place'],
        'Resort': ['resort', 'vacation spot', 'getaway', 'retreat'],
        'Landmark': ['landmark', 'attraction', 'monument', 'famous place', 'tourist spot', 'destination'],
        'Shopping/Restaurant': ['shopping', 'mall', 'shop', 'store', 'boutique', 'retail', 'shopping center', 'plaza'],
        'Spa/Restaurant': ['spa', 'massage', 'wellness', 'relaxation', 'treatment'],
        'Zoo': ['zoo', 'animal', 'wildlife', 'safari', 'animals'],
        'Farm/Restaurant': ['farm', 'agriculture', 'ranch', 'plantation', 'orchard', 'farm to table'],
        'Food Shop': ['food shop', 'bakery', 'pastry', 'dessert', 'sweets', 'delicatessen', 'specialty food'],
        'Beach Resort': ['beach resort', 'seaside resort', 'coastal resort', 'beachfront'],
        'Beach': ['beach', 'shore', 'coast', 'seaside', 'sand', 'bay'],
        'Natural Attraction': ['natural attraction', 'nature', 'scenery', 'landscape', 'waterfall', 'falls', 'cave', 'rock formation'],
        'Religious Site': ['religious site', 'church', 'temple', 'mosque', 'shrine', 'cathedral', 'chapel', 'monastery'],
        'Sports Facility': ['sports', 'athletics', 'stadium', 'arena', 'court', 'field', 'gym', 'fitness'],
        'Golf Course': ['golf', 'golf course', 'country club'],
        'Park': ['park', 'plaza', 'square', 'gardens', 'playground', 'recreation area'],
        'Mountain': ['mountain', 'hill', 'peak', 'summit', 'highlands', 'volcano', 'cliff', 'ridge'],
        'Hotel': ['hotel', 'inn', 'lodge', 'motel', 'accommodation', 'place to stay'],
        'Museum': ['museum', 'gallery', 'exhibit', 'collection', 'artifacts', 'heritage center'],
        'Garden': ['garden', 'botanical', 'flowers', 'plants', 'greenhouse'],
        'Accommodation': ['accommodation', 'lodging', 'stay', 'room', 'boarding'],
        'Historical Site': ['historical site', 'historic', 'heritage', 'ancient', 'ruins', 'archaeology', 'old'],
        'Hotel & Resort': ['hotel and resort', 'hotel & resort', 'resort hotel', 'luxury resort'],
        'Leisure': ['leisure', 'entertainment', 'recreation', 'fun', 'relaxation', 'pastime', 'amusement'],
        'Café/Restaurant': ['café', 'cafe', 'coffee shop', 'coffee', 'tea house', 'patisserie'],
        'Farm': ['farm', 'ranch', 'plantation', 'orchard', 'agricultural', 'dairy'],
        'Cafe': ['cafe', 'café', 'coffee shop', 'coffee house', 'espresso', 'cafeteria']
    }
    
    # Direct match - highest priority
    for category, synonyms in category_synonyms.items():
        if category.lower() in query_lower:
            return category
        
        for synonym in synonyms:
            if synonym in query_lower:
                return category
                
    # Check for partial matches in the available categories
    if available_categories:
        for category in available_categories:
            if category.lower() in query_lower:
                return category
    
    # Use best fuzzy match if no direct match found
    from difflib import SequenceMatcher
    best_match = None
    best_ratio = 0.6  # Threshold for considering a match
    
    # Try to match against synonyms
    for category, synonyms in category_synonyms.items():
        # Check category name directly
        ratio = SequenceMatcher(None, category.lower(), query_lower).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = category
            
        # Check all synonyms
        for synonym in synonyms:
            for word in query_lower.split():
                ratio = SequenceMatcher(None, synonym, word).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = category
    
    return best_match

# Internal recommendation function
def recommend_internal(query, city=None, category=None, limit=5):
    """
    Enhanced recommendation function with improved accuracy and context awareness
    """
    # Check if the query is just a location name (single word or short phrase)
    query_words = query.strip().split()
    is_location_query = len(query_words) <= 3
    
    available_cities = df['city'].unique().tolist() if df is not None else []
    
    # Normalize specialized category variations 
    query_lower = query.lower().strip()
    
    # Category normalization dictionary - maps variations to standard categories
    category_normalizations = {
        # Cafe/Coffee categories
        'cafe': 'Cafe', 'café': 'Cafe', 'coffee': 'Cafe', 'coffee shop': 'Cafe', 'cafeteria': 'Cafe',
        
        # Food Shop categories
        'food shop': 'Food Shop', 'eatery': 'Food Shop', 'food stall': 'Food Shop', 'food court': 'Food Shop',
        
        # Restaurant categories
        'restaurant': 'Restaurant', 'dining': 'Restaurant', 'bistro': 'Restaurant', 'diner': 'Restaurant',
        
        # Shopping categories
        'shopping': 'Shopping/Restaurant', 'mall': 'Shopping/Restaurant', 'shop': 'Shopping/Restaurant',
        
        # Resort/Hotel categories
        'resort': 'Resort', 'beach resort': 'Beach Resort', 'hotel & resort': 'Hotel & Resort',
        'hotel resort': 'Hotel & Resort', 'accommodation': 'Accommodation', 'hotel': 'Hotel',
        'restaurant/resort': 'Restaurant/Resort', 'stay': 'Accommodation',
        
        # Religious/Historical sites
        'church': 'Religious Site', 'temple': 'Religious Site', 'mosque': 'Religious Site', 'shrine': 'Religious Site',
        'historical': 'Historical Site', 'heritage': 'Historical Site', 'museum': 'Museum', 'landmark': 'Landmark',
        'historical place': 'Historical Site', 'historic': 'Historical Site', 'monument': 'Historical Site',
        
        # Nature categories
        'beach': 'Beach', 'mountain': 'Mountain', 'park': 'Park', 'garden': 'Garden', 
        'nature': 'Natural Attraction', 'falls': 'Natural Attraction', 'waterfall': 'Natural Attraction',
        'natural': 'Natural Attraction', 'scenery': 'Natural Attraction', 'outdoor': 'Natural Attraction',
        'landscape': 'Natural Attraction', 'view': 'Natural Attraction',
        
        # Leisure/Recreation
        'spa': 'Spa/Restaurant', 'farm': 'Farm', 'zoo': 'Zoo', 'sports': 'Sports Facility',
        'golf': 'Golf Course', 'golf course': 'Golf Course', 'leisure': 'Leisure',
        'recreation': 'Leisure', 'entertainment': 'Leisure', 'amusement': 'Leisure',
        
        # Mixed categories
        'spa/restaurant': 'Spa/Restaurant', 'farm/restaurant': 'Farm/Restaurant',
        'café/restaurant': 'Café/Restaurant', 'cafe restaurant': 'Café/Restaurant'
    }
    
    # Check for specialized category in query
    detected_category_type = None
    for keyword, normalized_category in category_normalizations.items():
        if keyword in query_lower:
            category = normalized_category
            detected_category_type = keyword
            logger.info(f"Detected category keyword: '{keyword}' → '{normalized_category}'")
            break
    
    # Check for specific city-category relationships
    if 'imus' in query_lower and ('cafe' in query_lower or 'coffee' in query_lower):
        city = 'Imus'
        category = 'Cafe'
    
    if 'tagaytay' in query_lower and ('food' in query_lower or 'restaurant' in query_lower):
        city = 'Tagaytay'
        category = 'Restaurant'
    
    if 'kawit' in query_lower and ('religious' in query_lower or 'church' in query_lower):
        city = 'Kawit'
        category = 'Religious Site'
    
    # Enhanced city detection with better handling of Cavite cities
    cavite_cities = {
        'kawit': 'Kawit', 
        'tagaytay': 'Tagaytay', 
        'amadeo': 'Amadeo', 
        'indang': 'Indang', 
        'ternate': 'Ternate',
        'maragondon': 'Maragondon', 
        'mendez': 'Mendez', 
        'alfonso': 'Alfonso', 
        'silang': 'Silang', 
        'imus': 'Imus',
        'bailen': 'Bailen', 
        'laurel': 'Laurel', 
        'dasmarinas': 'Dasmarinas', 
        'bacoor': 'Bacoor',
        'trece martires': 'Trece Martires', 
        'tanza': 'Tanza', 
        'naic': 'Naic', 
        'rosario': 'Rosario',
        'general trias': 'General Trias',
        'cavite city': 'Cavite City'
    }
    
    # Check for direct city mentions with proper capitalization
    for city_variation, normalized_city in cavite_cities.items():
        if city_variation in query_lower:
            city = normalized_city
            break
    
    # Enhanced city matching with improved fuzzy matching
    if is_location_query and not city:
        # Try exact match first with normalization
        normalized_query = ''.join(e for e in query_lower if e.isalnum())
        for available_city in available_cities:
            normalized_city = ''.join(e for e in available_city.lower() if e.isalnum())
            if normalized_city == normalized_query:
                logger.info(f"Found exact city match: {available_city}")
                city = available_city
                break
        
        # If no exact match, try word-by-word matching
        if not city:
            query_words = set(query_lower.split())
            for available_city in available_cities:
                city_words = set(available_city.lower().split())
                if query_words.intersection(city_words):
                    word_match_ratio = len(query_words.intersection(city_words)) / len(query_words)
                    if word_match_ratio >= 0.5:  # At least half of the words match
                        logger.info(f"Found word match: {available_city} with ratio {word_match_ratio}")
                        city = available_city
                        break
        
        # If still no match, try fuzzy matching with improved algorithm
        if not city:
            from difflib import SequenceMatcher
            best_match = None
            best_ratio = 0.7  # Increased threshold for better accuracy
            
            for available_city in available_cities:
                # Try both full name and parts of the name
                city_parts = available_city.lower().split()
                query_parts = query_lower.split()
                
                # Check full name match with normalization
                normalized_city = ''.join(e for e in available_city.lower() if e.isalnum())
                normalized_query = ''.join(e for e in query_lower if e.isalnum())
                full_ratio = SequenceMatcher(None, normalized_query, normalized_city).ratio()
                
                if full_ratio > best_ratio:
                    best_ratio = full_ratio
                    best_match = available_city
                
                # Check parts match with improved scoring
                for city_part in city_parts:
                    for query_part in query_parts:
                        normalized_city_part = ''.join(e for e in city_part if e.isalnum())
                        normalized_query_part = ''.join(e for e in query_part if e.isalnum())
                        part_ratio = SequenceMatcher(None, normalized_query_part, normalized_city_part).ratio()
                        
                        # Give higher weight to longer matches
                        weighted_ratio = part_ratio * (len(normalized_query_part) / len(normalized_query))
                        if weighted_ratio > best_ratio:
                            best_ratio = weighted_ratio
                            best_match = available_city
            
            if best_match:
                logger.info(f"Found fuzzy city match: {best_match} with ratio {best_ratio}")
                city = best_match
    
    # Enhanced category detection for simple location queries
    if is_location_query and city:
        detected_city = city
        detected_category = None
        budget = None
        sentiment_info = None
        budget_amount = None
        cleaned_query = query
    else:
        try:
            available_categories = df['category'].unique().tolist() if df is not None else []
            
            # Enhanced query understanding with context
            if not city:
                detected_city, detected_category, budget, cleaned_query, sentiment_info, budget_amount = extract_query_info(
                    query,
                    available_cities,
                    available_categories,
                    None
                )
                
                # Validate detected city
                if detected_city:
                    # Try exact match first
                    exact_match = next((c for c in available_cities if c.lower() == detected_city.lower()), None)
                    if exact_match:
                        detected_city = exact_match
                    else:
                        # Try partial match
                        partial_match = next((c for c in available_cities if detected_city.lower() in c.lower() or c.lower() in detected_city.lower()), None)
                        if partial_match:
                            detected_city = partial_match
                        else:
                            # Try fuzzy match as last resort
                            from difflib import SequenceMatcher
                            best_match = None
                            best_ratio = 0.6
                            for available_city in available_cities:
                                ratio = SequenceMatcher(None, detected_city.lower(), available_city.lower()).ratio()
                                if ratio > best_ratio:
                                    best_ratio = ratio
                                    best_match = available_city
                            if best_match:
                                detected_city = best_match
            
            city = detected_city or city
            category = detected_category or category
        except Exception as e:
            logger.error(f"Error in extract_query_info: {e}")
            budget = None
            sentiment_info = None
            budget_amount = None
            cleaned_query = query
    
    detected_city = city
    detected_category = category
    
    logger.info(f"Detected city: {detected_city}, category: {detected_category}")
    
    # If no category is detected from the normalization dictionary, try the more robust function
    if not detected_category_type and not category:
        available_categories = df['category'].unique().tolist() if df is not None else []
        potential_category = find_best_category_match(query_lower, available_categories)
        if potential_category:
            category = potential_category
            logger.info(f"Used advanced category matching to detect: {category}")
    
    # Check if the city-category combination exists in the dataset first
    if detected_city and detected_category and df is not None:
        city_category_mask = (
            (df['city'].str.lower() == detected_city.lower()) & 
            (df['category'].str.lower() == detected_category.lower())
        )
        
        if not city_category_mask.any():
            # Get available categories for this city
            city_mask = df['city'].str.lower() == detected_city.lower()
            if city_mask.any():
                available_categories_in_city = df[city_mask]['category'].unique().tolist()
                return {
                    "is_conversation": True,
                    "message": f"{detected_city} has no {detected_category} categories in our dataset. Available categories in {detected_city}: {', '.join(available_categories_in_city)}",
                    "detected_city": detected_city,
                    "detected_category": detected_category,
                    "available_categories": available_categories_in_city,
                    "data_availability": {
                        "city_exists": True,
                        "category_exists": detected_category in df['category'].str.lower().values,
                        "combination_exists": False
                    }
                }
            else:
                return {
                    "is_conversation": True,
                    "message": f"Sorry, I don't have data for {detected_city}. Available cities include: {', '.join(df['city'].unique()[:5])}{'...' if len(df['city'].unique()) > 5 else ''}",
                    "detected_city": detected_city,
                    "detected_category": detected_category,
                    "available_cities": df['city'].unique().tolist(),
                    "data_availability": {
                        "city_exists": False,
                        "category_exists": detected_category in df['category'].str.lower().values if detected_category else None,
                        "combination_exists": False
                    }
                }
    
    # Enhanced model-based recommendations with improved filtering
    destinations = []
    if model is not None and embeddings is not None and len(embeddings) > 0 and df is not None:
        try:
            logger.info("Attempting to use model-based recommendations...")
            # Enhanced query encoding
            query_encoding = tokenizer(
                query,
                add_special_tokens=True,
                max_length=512,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            ).to(device)
            
            # Get query embedding with improved attention
            with torch.no_grad():
                outputs = model.roberta(
                    input_ids=query_encoding['input_ids'],
                    attention_mask=query_encoding['attention_mask']
                )
                query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Calculate enhanced similarity scores
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            
            # Apply enhanced filters with scoring
            filtered_indices = list(range(len(df)))
            filtered_scores = []
            
            if detected_city:
                # Try exact city match first
                city_mask = df['city'].str.lower() == detected_city.lower()
                city_indices = [i for i in filtered_indices if i < len(city_mask) and city_mask.iloc[i]]
                
                # If no exact matches, try nearby cities or partial matches
                if not city_indices:
                    # Try partial matches with improved location relevance
                    city_indices = []
                    for idx in filtered_indices:
                        if idx >= len(df):
                            continue
                            
                        location_score = 0
                        current_city = df.iloc[idx]['city'].lower()
                        
                        # Exact city name match
                        if current_city == detected_city.lower():
                            location_score = 1.0
                        # City name contains query or vice versa
                        elif detected_city.lower() in current_city or current_city in detected_city.lower():
                            location_score = 0.8
                        # Check province match if available
                        elif 'province' in df.columns and df.iloc[idx]['province']:
                            province = df.iloc[idx]['province'].lower()
                            if detected_city.lower() in province:
                                location_score = 0.6
                            # Check for nearby cities in same province
                            elif any(city.lower() in province for city in available_cities):
                                location_score = 0.4
                        
                        if location_score > 0:
                            city_indices.append((idx, location_score))
                    
                    # Sort by location score and take top matches
                    if city_indices:
                        city_indices = sorted(city_indices, key=lambda x: x[1], reverse=True)
                        city_indices = [idx for idx, _ in city_indices[:limit*2]]  # Get more for filtering
                        logger.info(f"Found {len(city_indices)} places in related areas")
                
                if city_indices:
                    filtered_indices = city_indices
                    logger.info(f"Filtered to {len(filtered_indices)} places in/near {detected_city}")
            
            # Enhanced scoring system with location and category relevance
            for idx in filtered_indices:
                if idx >= len(df):
                    continue
                    
                base_score = similarities[idx]
                location_multiplier = 1.0
                
                # Location relevance scoring with enhanced province-city relationship
                if detected_city:
                    current_city = df.iloc[idx]['city'].lower()
                    current_province = df.iloc[idx]['province'].lower() if 'province' in df.columns else ''
                    
                    # Exact city match gets highest boost
                    if current_city == detected_city.lower():
                        location_multiplier = 2.5
                    # City is in the same province (Cavite)
                    elif current_province == 'cavite':
                        cavite_cities = [
                            'kawit', 'tagaytay', 'amadeo', 'indang', 'ternate', 
                            'maragondon', 'mendez', 'alfonso', 'silang', 'imus',
                            'bailen', 'laurel', 'dasmarinas', 'bacoor', 
                            'trece martires', 'tanza', 'naic', 'rosario', 
                            'general trias'
                        ]
                        # If detected city is in Cavite, boost other Cavite cities
                        if detected_city.lower() in cavite_cities:
                            location_multiplier = 2.0
                        # If detected city contains province name
                        elif 'cavite' in detected_city.lower():
                            location_multiplier = 1.8
                    # Partial city match gets medium boost
                    elif detected_city.lower() in current_city or current_city in detected_city.lower():
                        location_multiplier = 1.5
                    # Province match gets small boost
                    elif current_province and detected_city.lower() in current_province:
                        location_multiplier = 1.2
                
                # Apply location multiplier
                base_score *= location_multiplier
                
                # Category relevance scoring
                if detected_category and df.iloc[idx]['category'].lower() == detected_category.lower():
                    base_score *= 1.3
                
                # Consider ratings and popularity
                if 'ratings' in df.columns and df.iloc[idx]['ratings'] is not None:
                    try:
                        rating = float(df.iloc[idx]['ratings'])
                        base_score *= (1 + (rating / 10))
                    except (ValueError, TypeError):
                        pass
                
                if 'popularity_score' in df.columns and df.iloc[idx]['popularity_score'] is not None:
                    try:
                        popularity = float(df.iloc[idx]['popularity_score'])
                        base_score *= (1 + (popularity / 100))
                    except (ValueError, TypeError):
                        pass
                
                filtered_scores.append((idx, base_score))
            
            # Get top matches with enhanced scoring
            if filtered_scores:
                top_indices = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:limit*5]  # get more for filtering
                destinations = [df.iloc[idx].to_dict() for idx, _ in top_indices]
                logger.info(f"Found {len(destinations)} destinations using enhanced model-based recommendations")
                # STRICT CATEGORY FILTERING
                if detected_category:
                    before = len(destinations)
                    destinations = [d for d in destinations if d.get('category', '').lower() == detected_category.lower()]
                    logger.info(f"Filtered to {len(destinations)} destinations strictly in category '{detected_category}' (from {before})")
                destinations = destinations[:limit]  # limit after filtering
        except Exception as e:
            logger.error(f"Error using model-based recommendations: {e}")
            destinations = []
    
    # Enhanced fallback to database search with improved filtering
    if not destinations:
        logger.info("Falling back to enhanced database search...")
        try:
            # Try to get destinations with filters
            if detected_city or detected_category:
                # Enhanced multi-category search based on detected category
                if detected_category:
                    logger.info(f"Using enhanced multi-category search for {detected_category}")
                    
                    # For complex category groups, define related categories to search across
                    category_groups = {
                        'Cafe': ['Cafe', 'Café/Restaurant', 'Restaurant'],
                        'Restaurant': ['Restaurant', 'Café/Restaurant', 'Food Shop'],
                        'Café/Restaurant': ['Café/Restaurant', 'Cafe', 'Restaurant'],
                        'Hotel': ['Hotel', 'Hotel & Resort', 'Accommodation'],
                        'Resort': ['Resort', 'Hotel & Resort', 'Beach Resort'],
                        'Beach Resort': ['Beach Resort', 'Resort', 'Hotel & Resort'],
                        'Beach': ['Beach', 'Beach Resort', 'Natural Attraction'],
                        'Historical Site': ['Historical Site', 'Landmark', 'Museum'],
                        'Natural Attraction': ['Natural Attraction', 'Park', 'Mountain', 'Beach'],
                        'Shopping/Restaurant': ['Shopping/Restaurant', 'Restaurant', 'Leisure']
                    }
                    
                    # Check if the category belongs to a defined group
                    category_search_list = []
                    primary_weighting = 2.0  # Primary category gets higher weight
                    secondary_weighting = 1.0  # Related categories get normal weight
                    
                    if detected_category in category_groups:
                        category_search_list = category_groups[detected_category]
                        logger.info(f"Using category group: {category_search_list}")
                    else:
                        # Just use the detected category if no group defined
                        category_search_list = [detected_category]
                        logger.info(f"Using single category: {detected_category}")
                    
                    # Search across all related categories
                    all_category_destinations = []
                    scored_destinations = []
                    
                    for i, search_category in enumerate(category_search_list):
                        # Give highest weight to the primary category (first in list)
                        category_weight = primary_weighting if i == 0 else secondary_weighting
                        
                        # Get results for this category
                        category_results = db.get_destinations(
                            limit=limit * 2,  # Get more for good scoring
                            city=detected_city,
                            category=search_category
                        )
                        
                        logger.info(f"Found {len(category_results)} results for {search_category}")
                        
                        # Score the results based on category relevance and other factors
                        for dest in category_results:
                            score = category_weight  # Base score from category weight
                            
                            # Location relevance 
                            if detected_city and dest.get('city', '').lower() == detected_city.lower():
                                score *= 1.5
                            
                            # Exact category match gets priority
                            if dest.get('category') == detected_category:
                                score *= 1.3
                            
                            # Rating boost
                            if 'ratings' in dest and dest['ratings'] is not None:
                                try:
                                    rating = float(dest['ratings'])
                                    score *= (1 + (rating / 10))
                                except (ValueError, TypeError):
                                    pass
                            
                            # Description relevance to query
                            if 'description' in dest and dest['description']:
                                desc_relevance = 0
                                # Check for query terms in description
                                for term in query_lower.split():
                                    if len(term) > 3 and term in dest['description'].lower():
                                        desc_relevance += 0.2
                                score *= (1 + min(desc_relevance, 1.0))
                            
                            scored_destinations.append((dest, score))
                    
                    # Remove duplicates, keeping highest score
                    unique_destinations = {}
                    for dest, score in scored_destinations:
                        dest_id = dest.get('id', dest.get('name', ''))
                        if dest_id not in unique_destinations or score > unique_destinations[dest_id][1]:
                            unique_destinations[dest_id] = (dest, score)
                    
                    # Sort by score and take top results
                    if unique_destinations:
                        sorted_destinations = sorted(
                            unique_destinations.values(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )
                        destinations = [dest for dest, _ in sorted_destinations[:limit]]
                        logger.info(f"Found {len(destinations)} destinations across related categories")
                
                # If no results with the multi-category approach or if no category was detected
                if not destinations:
                    logger.info("Trying direct filters...")
                    # Try exact city and category match
                    destinations = db.get_destinations(
                        limit=limit * 2,
                        city=detected_city,
                        category=detected_category
                    )
                    logger.info(f"Found {len(destinations)} destinations with exact filters")
                
                # If no results with exact city match, try partial matches
                if not destinations and detected_city:
                    logger.info("Trying partial city matches...")
                    all_destinations = db.get_destinations(limit=200)  # Get more destinations to filter
                    
                    # Filter for partial matches with scoring
                    scored_destinations = []
                    for dest in all_destinations:
                        score = 0
                        dest_city = dest.get('city', '').lower()
                        dest_province = dest.get('province', '').lower()
                        
                        # Exact city match
                        if dest_city == detected_city.lower():
                            score = 1.0
                        # City is in Cavite province
                        elif dest_province == 'cavite':
                            cavite_cities = [
                                'kawit', 'tagaytay', 'amadeo', 'indang', 'ternate', 
                                'maragondon', 'mendez', 'alfonso', 'silang', 'imus',
                                'bailen', 'laurel', 'dasmarinas', 'bacoor', 
                                'trece martires', 'tanza', 'naic', 'rosario', 
                                'general trias'
                            ]
                            # If detected city is in Cavite, boost other Cavite cities
                            if detected_city.lower() in cavite_cities:
                                score = 0.9
                            # If detected city contains province name
                            elif 'cavite' in detected_city.lower():
                                score = 0.8
                        # City contains query or vice versa
                        elif detected_city.lower() in dest_city or dest_city in detected_city.lower():
                            score = 0.7
                        # Province match
                        elif dest_province and detected_city.lower() in dest_province:
                            score = 0.6
                        # Nearby city in same province
                        elif dest_province and any(city.lower() in dest_province for city in available_cities):
                            score = 0.4
                        
                        # Apply category boost if specified
                        if detected_category and dest.get('category', '').lower() == detected_category.lower():
                            score *= 1.3
                        
                        # Apply rating boost
                        if 'ratings' in dest and dest['ratings'] is not None:
                            try:
                                rating = float(dest['ratings'])
                                score *= (1 + (rating / 10))
                            except (ValueError, TypeError):
                                pass
                        
                        if score > 0:
                            scored_destinations.append((dest, score))
                    
                    # Sort by score and take top results
                    if scored_destinations:
                        scored_destinations.sort(key=lambda x: x[1], reverse=True)
                        destinations = [dest for dest, _ in scored_destinations[:limit]]
                        logger.info(f"Found {len(destinations)} destinations with partial location match")
                
                # If still no results and we have both filters, try with just one
                if not destinations and detected_city and detected_category:
                    # Try with just city
                    destinations = db.get_destinations(limit=limit, city=detected_city)
                    if destinations:
                        logger.info(f"Found {len(destinations)} destinations in {detected_city}")
                    else:
                        # Try with just category
                        destinations = db.get_destinations(limit=limit, category=detected_category)
                        if destinations:
                            logger.info(f"Found {len(destinations)} {detected_category} destinations")
                
                # If no results with filters, try full-text search with location boost
                if not destinations:
                    search_query = query
                    if detected_city:
                        search_query = f"{detected_city} {query}"
                    all_results = db.search_destinations(search_query, limit=limit * 2)
                    
                    # Score and sort results
                    scored_results = []
                    for dest in all_results:
                        score = 1.0
                        
                        # Location relevance
                        if detected_city:
                            dest_city = dest.get('city', '').lower()
                            if dest_city == detected_city.lower():
                                score *= 2.0
                            elif detected_city.lower() in dest_city or dest_city in detected_city.lower():
                                score *= 1.5
                            elif dest.get('province', '').lower() and detected_city.lower() in dest.get('province', '').lower():
                                score *= 1.2
                        
                        # Category relevance
                        if detected_category and dest.get('category', '').lower() == detected_category.lower():
                            score *= 1.3
                        
                        # Rating boost
                        if 'ratings' in dest and dest['ratings'] is not None:
                            try:
                                rating = float(dest['ratings'])
                                score *= (1 + (rating / 10))
                            except (ValueError, TypeError):
                                pass
                        
                        scored_results.append((dest, score))
                    
                    # Sort by score and take top results
                    if scored_results:
                        scored_results.sort(key=lambda x: x[1], reverse=True)
                        destinations = [dest for dest, _ in scored_results[:limit]]
                logger.info(f"Found {len(destinations)} destinations with full-text search")
                    
            # Last resort: get random destinations with location preference
            if not destinations:
                all_destinations = db.get_destinations(limit=100)
                if detected_city:
                    # Score destinations by location relevance
                    scored_destinations = []
                    for dest in all_destinations:
                        score = 0
                        dest_city = dest.get('city', '').lower()
                        dest_province = dest.get('province', '').lower()
                        
                        if dest_city == detected_city.lower():
                            score = 1.0
                        elif detected_city.lower() in dest_city or dest_city in detected_city.lower():
                            score = 0.8
                        elif dest_province and detected_city.lower() in dest_province:
                            score = 0.6
                        
                        if score > 0:
                            scored_destinations.append((dest, score))
                    
                    if scored_destinations:
                        scored_destinations.sort(key=lambda x: x[1], reverse=True)
                        destinations = [dest for dest, _ in scored_destinations[:limit]]
                    else:
                        destinations = all_destinations[:limit]
                else:
                    destinations = all_destinations[:limit]
                
                logger.info(f"Found {len(destinations)} destinations as last resort")
                
        except Exception as e:
            logger.error(f"Error in database search: {e}")
            destinations = []
    
    # Enhanced recommendation formatting with more context
    recommendations = []
    for dest in destinations:
        try:
            # Enhanced rating conversion
            if 'ratings' in dest and dest['ratings'] is not None:
                raw_rating = float(dest['ratings'])
                if raw_rating > 5:
                    rating = min(5, max(1, int(raw_rating * 0.5)))
                else:
                    rating = min(5, max(1, int(raw_rating)))
            else:
                rating = 3
            
            # Enhanced recommendation object with more context
            recommendation = {
                'id': dest.get('id', ''),
                'name': dest.get('name', 'Unknown Destination'),
                'city': dest.get('city', 'Unknown City'),
                'province': dest.get('province', ''),
                'category': dest.get('category', 'Unknown Category'),
                'description': dest.get('description', 'No description available'),
                'rating': rating,
                'budget': dest.get('budget', 'medium'),
                'popularity_score': dest.get('popularity_score', 0),
                'last_updated': dest.get('last_updated', None),
                'verified': dest.get('verified', False)
            }
            
            # Add optional fields with validation
            for field in ['latitude', 'longitude', 'operating_hours', 'contact_information', 'metadata']:
                if field in dest and dest[field]:
                    recommendation[field] = dest[field]
        
            recommendations.append(recommendation)
        except Exception as e:
            logger.error(f"Error formatting recommendation: {e}")
            continue
    
    # Enhanced response for no results
    if detected_city and not recommendations:
        return {
            "is_conversation": True,
            "message": f"I couldn't find any places in {detected_city}. Would you like to try a different location or be more specific about what you're looking for?",
            "query_understanding": {
                "detected_city": detected_city,
                "detected_category": detected_category
            }
        }
    
    # Normalize detected category for display
    display_category = detected_category
    
    # When we have recommendations, try to extract the most common real category
    if recommendations:
        # Extract categories from results if detected_category was inferred
        if detected_category_type:
            # Count category occurrences in results
            category_counts = {}
            for rec in recommendations:
                cat = rec.get('category')
                if cat:
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Set display category to most common if we have a significant count
            if category_counts:
                most_common_category = max(category_counts.items(), key=lambda x: x[1])[0]
                # Only override if the detected category was not explicitly specified
                if not category and category_counts[most_common_category] >= len(recommendations) * 0.3:  # If at least 30% have this category
                    display_category = most_common_category
    
    # Log what we're returning
    logger.info(f"Returning {len(recommendations)} recommendations with detected_city: {detected_city}, category: {display_category}")
        
    return {
        "is_conversation": False,
        "detected_city": detected_city,
        "detected_category": display_category,
        "recommendations": recommendations,
        "query_understanding": {
            "detected_city": detected_city,
            "detected_category": display_category,
            "detected_category_type": detected_category_type,
            "original_query": query
        }
    }

# Route for recommendations
@bp.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Check for required components
        if model is None:
            logger.warning("Model not loaded, falling back to database search")
        
        # Get request data
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing query parameter',
                'is_conversation': True,
                'message': "I need to know what you're looking for. Could you please specify what type of places you're interested in?"
            }), 400
        
        query = data['query']
        session_id = data.get('session_id')
        rating_filter = data.get('rating')  # Get rating filter if present
        result_limit = data.get('limit', 5)  # Get limit from request, default to 5
        
        # Convert limit to integer and validate range
        try:
            result_limit = int(result_limit)
            # Set reasonable min and max boundaries
            if result_limit < 1:
                result_limit = 1
            elif result_limit > 20:  # Maximum 20 results
                result_limit = 20
        except (ValueError, TypeError):
            # If limit is not a valid integer, use default 5
            result_limit = 5
        
        if rating_filter:
            try:
                rating_filter = int(rating_filter)
                if rating_filter < 1 or rating_filter > 5:
                    rating_filter = None  # Reset if invalid range
            except (ValueError, TypeError):
                rating_filter = None  # Reset if not a valid integer
        
        # Check cache first
        cached_result = knowledge_cache.get(query)
        if cached_result and not rating_filter and result_limit == 5:  # Only use cache if no rating filter is applied and using default limit
            # Update session context from cached result if available
            if session_id and 'recommendations' in cached_result:
                update_session_context_from_recommendations(session_id, cached_result['recommendations'], query)
            
            return jsonify(cached_result)
        
        # Check if this is a conversation rather than a recommendation request
        conversation_result = handle_conversation(query, session_id)
        if conversation_result:
            return jsonify(conversation_result)
        
        # Enhanced query understanding
        query_understanding = understand_query(query, session_id)
        
        # Get conversation context if session_id is provided
        conversation_context = None
        if session_id:
            conversation_context = knowledge_cache.get_conversation_context(session_id, query)
        
        # Use context-aware recommendations
        city = query_understanding.get('city')
        category = query_understanding.get('category')
        
        # Internal recommendation function with advanced understanding
        result = recommend_internal(query, city, category, limit=result_limit)
        
        # Apply rating filter if specified
        if rating_filter and 'recommendations' in result:
            # Filter recommendations by minimum rating
            original_count = len(result['recommendations'])
            result['recommendations'] = [
                rec for rec in result['recommendations'] 
                if rec.get('rating', 0) >= rating_filter
            ]
            
            # Add rating filter info to the response
            result['rating_filter_applied'] = rating_filter
            
            # Add message if filters reduced results significantly
            if len(result['recommendations']) == 0 and original_count > 0:
                result['filter_message'] = f"All places were filtered out by the {rating_filter}+ star requirement."
            elif len(result['recommendations']) < 3 and original_count > 3:
                result['filter_message'] = f"Showing only {len(result['recommendations'])} places with {rating_filter}+ stars."
        
        # Generate follow-up suggestions based on recommendations
        if 'recommendations' in result and result['recommendations']:
            result['follow_up_suggestions'] = generate_follow_up_suggestions(
                result['recommendations'], 
                conversation_context
            )
            
            # Update session context with these recommendations
            if session_id:
                update_session_context_from_recommendations(session_id, result['recommendations'], query)
        
        # Cache the result
        knowledge_cache.add(query, result, session_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in /api/recommend: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': str(e),
            'is_conversation': True,
            'message': "I encountered an error while processing your request. Please try again with a different query."
        }), 500

@bp.route('/session/<session_id>', methods=['GET'])
def validate_session(session_id):
    """Endpoint to validate a session"""
    try:
        session = db.get_session(session_id)
        if session:
            return jsonify({"valid": True, "session_id": session_id}), 200
        else:
            return jsonify({"valid": False}), 404
    except Exception as e:
        logger.error(f"Error validating session: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route('/create_session', methods=['POST'])
def create_session():
    """Create a new session for tracking user interactions"""
    try:
        # Generate a new session ID
        result = db.create_session()
        
        if 'error' in result:
            return jsonify(result), 500
            
        session_id = result['session_id']
        
        # Initialize session in KnowledgeCache
        knowledge_cache.session_contexts[session_id] = {
            'created_at': time.time(),
            'preferences': {}
        }
        
        return jsonify({"session_id": session_id}), 201
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route('/categories', methods=['GET'])
def get_categories():
    """
    Get all distinct categories available in the database.
    
    Returns a list of category names.
    """
    try:
        # Get all distinct categories from the database
        categories = db.get_distinct_categories()
        
        return jsonify({
            "categories": categories
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving categories: {e}")
        return jsonify({"error": str(e)}), 500

# Add import of time module which is used in the update_session_context_from_recommendations function
import time 