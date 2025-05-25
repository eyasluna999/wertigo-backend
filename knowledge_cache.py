import time
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

# Expand the KnowledgeCache class with enhanced conversation context capabilities
class KnowledgeCache:
    def __init__(self, max_size=100, expiry_time=3600):  # 1 hour expiry
        self.cache = {}
        self.query_history = []
        self.max_size = max_size
        self.expiry_time = expiry_time
        self.category_preferences = defaultdict(Counter)
        self.city_preferences = defaultdict(Counter)
        self.session_contexts = {}
        self.conversation_history = defaultdict(list)
        self.sentiment_history = defaultdict(list)
        self.topic_transitions = defaultdict(list)
    
    def add(self, query, result, session_id=None):
        # Normalize query for better matching
        normalized_query = self._normalize_query(query)
        current_time = time.time()
        
        # If cache is full, remove oldest entry
        if len(self.cache) >= self.max_size:
            oldest_query = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_query]
        
        # Add to cache
        self.cache[normalized_query] = {
            'result': result,
            'timestamp': current_time,
            'hits': 1
        }
        
        # Track category and city preferences if available
        if isinstance(result, dict) and 'recommendations' in result:
            for rec in result.get('recommendations', []):
                if session_id:
                    if 'category' in rec:
                        self.category_preferences[session_id][rec['category']] += 1
                    if 'city' in rec:
                        self.city_preferences[session_id][rec['city']] += 1
        
        # Add to query history
        self.query_history.append({
            'query': query,
            'normalized': normalized_query,
            'timestamp': current_time,
            'session_id': session_id
        })
    
    def get(self, query):
        normalized_query = self._normalize_query(query)
        if normalized_query in self.cache:
            entry = self.cache[normalized_query]
            current_time = time.time()
            
            # Check if entry is expired
            if current_time - entry['timestamp'] > self.expiry_time:
                del self.cache[normalized_query]
                return None
            
            # Update hits and timestamp
            entry['hits'] += 1
            entry['timestamp'] = current_time
            return entry['result']
        
        return None
    
    def get_similar_query(self, query, threshold=0.8):
        """Find similar queries in cache above similarity threshold"""
        normalized_query = self._normalize_query(query)
        
        # For now use basic word overlap as similarity measure
        query_words = set(normalized_query.split())
        best_match = None
        best_score = threshold
        
        for cached_query in self.cache:
            cached_words = set(cached_query.split())
            
            # Calculate Jaccard similarity
            if not query_words or not cached_words:
                continue
                
            intersection = len(query_words.intersection(cached_words))
            union = len(query_words.union(cached_words))
            
            score = intersection / union if union > 0 else 0
            
            if score > best_score:
                best_score = score
                best_match = cached_query
        
        if best_match:
            return self.cache[best_match]['result']
        
        return None
    
    def update_session_context(self, session_id, context_data):
        """Update context for a specific session"""
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = {}
        
        self.session_contexts[session_id].update(context_data)
    
    def get_session_context(self, session_id):
        """Get context for a specific session"""
        return self.session_contexts.get(session_id, {})
    
    def get_preferred_categories(self, session_id, top_n=3):
        """Get preferred categories for a session"""
        if session_id not in self.category_preferences:
            return []
        
        return [cat for cat, _ in self.category_preferences[session_id].most_common(top_n)]
    
    def get_preferred_cities(self, session_id, top_n=3):
        """Get preferred cities for a session"""
        if session_id not in self.city_preferences:
            return []
        
        return [city for city, _ in self.city_preferences[session_id].most_common(top_n)]
    
    def _normalize_query(self, query):
        """Normalize query for better matching"""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Remove extra whitespaces
        query = " ".join(query.split())
        
        # Remove common filler words for better matching
        filler_words = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "is", "are", "was", "were"}
        query_words = query.split()
        query_words = [word for word in query_words if word not in filler_words]
        
        return " ".join(query_words)
    
    def add_to_conversation(self, session_id, user_message, system_response, sentiment=None):
        """Add a conversation exchange to the history"""
        if not session_id:
            return
            
        # Get timestamp
        timestamp = time.time()
        
        # Extract topics from the message
        topics = self._extract_topics(user_message)
        
        # Analyze sentiment if not provided
        if sentiment is None:
            sentiment = self._analyze_sentiment(user_message)
        
        # Create conversation entry
        conversation_entry = {
            'timestamp': timestamp,
            'user_message': user_message,
            'system_response': system_response,
            'topics': topics,
            'sentiment': sentiment
        }
        
        # Add to conversation history
        self.conversation_history[session_id].append(conversation_entry)
        
        # Track sentiment over time
        self.sentiment_history[session_id].append({
            'timestamp': timestamp,
            'sentiment': sentiment
        })
        
        # Track topic transitions
        if len(self.conversation_history[session_id]) > 1:
            prev_entry = self.conversation_history[session_id][-2]
            prev_topics = prev_entry.get('topics', [])
            
            # Record the transition between topics
            for prev_topic in prev_topics:
                for curr_topic in topics:
                    if prev_topic != curr_topic:
                        self.topic_transitions[session_id].append({
                            'from': prev_topic,
                            'to': curr_topic,
                            'timestamp': timestamp
                        })
    
    def get_conversation_summary(self, session_id, max_entries=5):
        """Get a summary of recent conversation"""
        if not session_id or session_id not in self.conversation_history:
            return []
            
        # Get most recent entries
        recent_entries = self.conversation_history[session_id][-max_entries:]
        
        # Create summary
        summary = []
        for entry in recent_entries:
            summary.append({
                'user_message': entry['user_message'],
                'topics': entry['topics'],
                'sentiment': entry['sentiment']
            })
            
        return summary
    
    def get_frequently_discussed_topics(self, session_id, top_n=3):
        """Get the most frequently discussed topics in a session"""
        if not session_id or session_id not in self.conversation_history:
            return []
            
        # Collect all topics
        all_topics = []
        for entry in self.conversation_history[session_id]:
            all_topics.extend(entry.get('topics', []))
            
        # Count frequencies
        topic_counter = Counter(all_topics)
        
        # Return top N
        return [topic for topic, _ in topic_counter.most_common(top_n)]
    
    def get_conversation_context(self, session_id, query=None):
        """Get enhanced conversation context for better response generation"""
        if not session_id:
            return {}
            
        context = {
            'session_id': session_id,
            'preferences': {}
        }
        
        # Get basic session context
        session_context = self.get_session_context(session_id)
        context.update(session_context)
        
        # Add category preferences
        if session_id in self.category_preferences:
            preferred_categories = [cat for cat, _ in self.category_preferences[session_id].most_common(3)]
            context['preferences']['categories'] = preferred_categories
            
        # Add city preferences
        if session_id in self.city_preferences:
            preferred_cities = [city for city, _ in self.city_preferences[session_id].most_common(3)]
            context['preferences']['cities'] = preferred_cities
        
        # Add conversation history summary
        if session_id in self.conversation_history:
            context['conversation'] = {
                'exchanges': len(self.conversation_history[session_id]),
                'recent_topics': self.get_frequently_discussed_topics(session_id)
            }
            
            # Add recent sentiment trend
            if session_id in self.sentiment_history and len(self.sentiment_history[session_id]) > 0:
                recent_sentiments = [entry['sentiment'] for entry in self.sentiment_history[session_id][-3:]]
                avg_sentiment = sum(recent_sentiments) / len(recent_sentiments) if recent_sentiments else 0
                context['conversation']['sentiment_trend'] = avg_sentiment
        
        # If query is provided, analyze it for contextual relevance
        if query:
            context['current_query'] = {
                'topics': self._extract_topics(query),
                'sentiment': self._analyze_sentiment(query)
            }
            
            # Check for topic continuity
            if 'conversation' in context and 'recent_topics' in context['conversation']:
                query_topics = self._extract_topics(query)
                recent_topics = context['conversation']['recent_topics']
                context['current_query']['topic_continuity'] = any(topic in recent_topics for topic in query_topics)
        
        return context
    
    def _extract_topics(self, text):
        """Extract topics from a text"""
        # Simple topic extraction based on keywords
        travel_topics = {
            'destination': ['city', 'destination', 'place', 'location', 'country', 'visit'],
            'accommodation': ['hotel', 'resort', 'stay', 'accommodation', 'room', 'booking'],
            'transportation': ['flight', 'train', 'bus', 'car', 'transportation', 'travel'],
            'activities': ['activity', 'tour', 'sightseeing', 'adventure', 'experience'],
            'food': ['food', 'restaurant', 'cuisine', 'eat', 'dining'],
            'budget': ['budget', 'cost', 'price', 'expensive', 'cheap', 'affordable'],
            'planning': ['plan', 'itinerary', 'schedule', 'booking', 'reservation'],
            'weather': ['weather', 'season', 'temperature', 'climate'],
            'cultural': ['culture', 'history', 'museum', 'traditional', 'local'],
            'nature': ['nature', 'beach', 'mountain', 'lake', 'park', 'hiking']
        }
        
        found_topics = []
        text_lower = text.lower()
        
        for topic, keywords in travel_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)
                
        return found_topics
    
    def _analyze_sentiment(self, text):
        """Analyze sentiment of a text (basic version)"""
        # Simple sentiment analysis based on keywords
        positive_keywords = ['good', 'great', 'excellent', 'amazing', 'love', 'enjoy', 'nice', 'happy', 'beautiful', 
                           'wonderful', 'perfect', 'best', 'like', 'recommend', 'fantastic', 'awesome']
        negative_keywords = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'poor', 'disappointing', 'worst',
                           'horrible', 'avoid', 'expensive', 'dirty', 'crowded', 'dangerous']
        
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        # Simple scoring between -1 and 1
        total = positive_count + negative_count
        if total == 0:
            return 0  # Neutral
            
        return (positive_count - negative_count) / total

# Initialize knowledge cache
knowledge_cache = KnowledgeCache() 