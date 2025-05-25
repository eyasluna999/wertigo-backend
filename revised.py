import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os
import re
import nltk
import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix
import seaborn as sns

# Force CPU usage
# Define device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')
print(f"Using device: {device}")

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.download('punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
    def analyze_sentiment(self, text):
        """
        Analyze sentiment using both TextBlob and VADER for more accurate results
        """
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity  # -1 to 1
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        
        # Extract sentiment-bearing words
        sentiment_words = self._extract_sentiment_words(text)
        
        # Determine overall sentiment
        if textblob_sentiment > 0.1:
            overall = 'positive'
        elif textblob_sentiment < -0.1:
            overall = 'negative'
        else:
            overall = 'neutral'
            
        return {
            'textblob_sentiment': textblob_sentiment,
            'vader_scores': vader_scores,
            'sentiment_words': sentiment_words,
            'overall_sentiment': overall
        }
    
    def _extract_sentiment_words(self, text):
        """
        Extract words that contribute to sentiment
        """
        words = word_tokenize(text.lower())
        sentiment_words = []
        
        for word in words:
            # Check VADER sentiment
            if abs(self.vader.polarity_scores(word)['compound']) > 0.1:
                sentiment_words.append(word)
                
        return sentiment_words

# Download spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import sys
    logger.info("Downloading spaCy model for NER...")
    os.system(f"{sys.executable} -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def get_device():
    """Get the device to use for computations"""
    return device

# Load the data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# Preprocess the data
def preprocess_data(df):
    # Fill NaN values
    df = df.fillna('')

    # Split combined categories and create a list of all categories
    df['all_categories'] = df['category'].apply(lambda x: [cat.strip() for cat in str(x).split(',')])

    # Create a combined text field with weighted importance
    # Description is repeated twice to give it more weight in the model
    df['combined_text'] = df['description'] + ' ' + df['description'] + ' ' + \
                         df['name'] + ' ' + \
                         df['category'] + ' ' + \
                         df['metadata']

    # Encode the categories - now using the first category as primary
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'].apply(lambda x: str(x).split(',')[0].strip()))

    # Save the label encoder mapping for future use
    category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    logger.info(f"Category mapping: {category_mapping}")

    return df, label_encoder

# Advanced function to extract query information using spaCy NER and pattern matching
def extract_query_info(query_text, available_cities, available_categories, available_budgets=None):
    """
    Extract city, category, and budget information from a user query using NER and pattern matching.
    """
    query_lower = query_text.lower()

    # Initialize variables
    extracted_city = None
    extracted_category = None
    extracted_budget = None
    budget_amount = None
    
    # Process with spaCy for NER
    doc = nlp(query_text)
    
    # Extract cities using spaCy NER
    for ent in doc.ents:
        if ent.label_ == "GPE" and not extracted_city:  # GPE = Geopolitical Entity
            potential_city = ent.text
            # Verify against our available cities
            for city in available_cities:
                if potential_city.lower() in city.lower() or city.lower() in potential_city.lower():
                    extracted_city = city
                    break
    
    # Backup method: Check direct city mentions if spaCy NER didn't find any
    if not extracted_city:
        for city in available_cities:
            city_lower = city.lower()
            if city_lower in query_lower:
                extracted_city = city
                break

    # Category mapping with synonyms and related terms
    category_mapping = {
        "hotel": ["hotel", "resort", "lodge", "inn", "hostel", "stay", "bed and breakfast", "guesthouse", "motel", "lodging", "room"],
        "cafe": ["cafe", "coffee", "restaurant", "breakfast", "lunch", "dinner"],
        "restaurant":["eat", "hungry"],
        "historical site": ["historical", "history", "heritage", "museum", "shrine", "ancient", "old", "traditional"],
        "natural attraction": ["nature", "natural", "outdoors", "mountain", "lake", "volcano", "falls", 
                               "waterfall", "beach", "ocean", "sea", "river", "hiking", "trek", "forest"],
        "leisure": ["park", "amusement", "rides", "attraction", "entertainment", "fun", "thrill"],
        "museum": ["museum", "collection", "exhibit", "gallery", "art", "cultural", "artifacts"],
        "beach resort": ["beach resort", "seaside resort", "coastal resort", "ocean resort", "beachfront resort"]
    }

    # Extract category using mapped terms
    for category in available_categories:
        category_lower = category.lower()
        # Direct match
        if category_lower in query_lower:
            extracted_category = category
            break

        # Check synonyms
        for cat, synonyms in category_mapping.items():
            if cat.lower() == category_lower:
                for synonym in synonyms:
                    if synonym in query_lower:
                        extracted_category = category
                        break
                if extracted_category:
                    break

    # Extract budget using regex patterns
    budget_patterns = [
        r'under\s*(\d+)\s*(?:pesos|PHP)?',  # "under 500 pesos"
        r'below\s*(\d+)\s*(?:pesos|PHP)?',  # "below 500 pesos"
        r'less than\s*(\d+)\s*(?:pesos|PHP)?',  # "less than 500 pesos"
        r'(\d+)\s*(?:pesos|PHP)?\s*or less',  # "500 pesos or less"
        r'budget of\s*(\d+)\s*(?:pesos|PHP)?',  # "budget of 500 pesos"
        r'(\d+)\s*(?:pesos|PHP)?',  # "500 pesos"
    ]

    for pattern in budget_patterns:
        match = re.search(pattern, query_text.lower())
        if match:
            budget_amount = int(match.group(1))
            break
    
    # Remove city, category, and budget mentions from query to get cleaner core query
    cleaned_query = query_text
    
    if extracted_city:
        cleaned_query = re.sub(r'\b' + re.escape(extracted_city) + r'\b', '', cleaned_query, flags=re.IGNORECASE)
    
    if extracted_category:
        cleaned_query = re.sub(r'\b' + re.escape(extracted_category) + r'\b', '', cleaned_query, flags=re.IGNORECASE)
        # Also remove synonyms
        for cat, synonyms in category_mapping.items():
            if cat.lower() == extracted_category.lower():
                for synonym in synonyms:
                    cleaned_query = re.sub(r'\b' + re.escape(synonym) + r'\b', '', cleaned_query, flags=re.IGNORECASE)

    # Clean up extra spaces
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()

    return extracted_city, extracted_category, extracted_budget, cleaned_query, None, budget_amount

# Create dataset class
class DestinationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Model definition
class DestinationRecommender(torch.nn.Module):
    def __init__(self, num_labels, dropout=0.5):
        super(DestinationRecommender, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.intermediate = torch.nn.Linear(self.roberta.config.hidden_size, self.roberta.config.hidden_size // 2)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size // 2, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        pooled_output = torch.relu(self.intermediate(pooled_output))
        pooled_output = self.dropout2(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Training function
def train_model(model, train_dataloader, val_dataloader, epochs=10, learning_rate=2e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2
    )
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # Lists to store metrics
    train_losses = []
    val_losses = []
    val_metrics = {
        'accuracy': [],
        'f1': [],
        'recall': [],
        'precision': []
    }

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for batch in train_progress_bar:
            # Move batch to GPU if available
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")

        with torch.no_grad():
            for batch in val_progress_bar:
                # Move batch to GPU if available
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                # Get predictions
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())  # Move predictions back to CPU for metrics
                all_labels.extend(labels.cpu().numpy())  # Move labels back to CPU for metrics

                val_progress_bar.set_postfix({'loss': loss.item()})

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')

        # Store metrics
        val_metrics['accuracy'].append(accuracy)
        val_metrics['f1'].append(f1)
        val_metrics['recall'].append(recall)
        val_metrics['precision'].append(precision)

        # Print metrics
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss/len(val_dataloader):.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        
        # Print detailed classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(all_labels, all_preds))

        # Update learning rate
        scheduler.step(val_loss)

    return model, train_losses, val_losses, val_metrics

# Function to create embeddings for destinations
def create_destination_embeddings(model, dataset, dataloader):
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Creating embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Get the CLS token embedding
            outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.extend(embeddings)

    # Ensure we have exactly the right number of embeddings
    if len(all_embeddings) > len(dataset):
        all_embeddings = all_embeddings[:len(dataset)]
    elif len(all_embeddings) < len(dataset):
        # Pad with zeros if we somehow have fewer embeddings
        padding = np.zeros((len(dataset) - len(all_embeddings), all_embeddings[0].shape[0]))
        all_embeddings = np.vstack([all_embeddings, padding])

    return np.array(all_embeddings), list(range(len(dataset)))

# Enhanced recommendation function with intelligent filtering including budget
def get_recommendations(query_text, tokenizer, model, embeddings, df, city=None, category=None, budget=None, budget_amount=None, top_n=5):
    """
    Get destination recommendations based on a query text and optional filters.
    Now includes numeric budget filtering and handles multiple categories.
    """
    # Tokenize the query
    query_encoding = tokenizer(
        query_text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    ).to(device)

    # Get the query embedding
    model.eval()
    with torch.no_grad():
        outputs = model.roberta(
            input_ids=query_encoding['input_ids'],
            attention_mask=query_encoding['attention_mask']
        )
        query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Create a series of similarities with df indices
    similarity_series = pd.Series(similarities, index=df.index)

    # Apply filters
    filter_applied = False
    filtered_df = df.copy()

    # Apply city filter if specified
    if city:
        city_mask = filtered_df['city'].str.lower() == city.lower()
        if not any(city_mask):
            logger.warning(f"No destinations found in city: {city}")
            return pd.DataFrame(), np.array([])
        filtered_df = filtered_df[city_mask]
        filter_applied = True

    # Apply category filter if specified
    if category:
        # Check if the category exists in any of the categories in all_categories
        category_mask = filtered_df['all_categories'].apply(
            lambda cats: any(cat.lower() == category.lower() for cat in cats)
        )
        if not any(category_mask):
            logger.warning(f"No destinations found with category: {category}")
            if filter_applied:
                return pd.DataFrame(), np.array([])
        else:
            filtered_df = filtered_df[category_mask]
            filter_applied = True
    
    # Apply budget filter if specified
    if budget_amount is not None:
        # Convert budget column to numeric, handling any non-numeric values
        filtered_df['budget'] = pd.to_numeric(filtered_df['budget'], errors='coerce')
        
        # Check if the query contains "under", "below", or "less than"
        is_strict_budget = any(word in query_text.lower() for word in ['under', 'below', 'less than'])
        
        if is_strict_budget:
            # For "under" queries, strictly enforce the budget limit
            budget_mask = filtered_df['budget'] <= budget_amount
        else:
            # For other queries, allow a small buffer (10% instead of 20%)
            budget_mask = filtered_df['budget'] <= (budget_amount * 1.1)
        
        # Log budget filtering details
        logger.info(f"Budget filtering: Amount={budget_amount}, Strict={is_strict_budget}")
        logger.info(f"Found {budget_mask.sum()} destinations within budget")
        
        if not any(budget_mask):
            logger.warning(f"No destinations found within budget: {budget_amount}")
            if filter_applied:
                return pd.DataFrame(), np.array([])
        else:
            filtered_df = filtered_df[budget_mask]
            filter_applied = True
        
        # Sort by budget within the filtered results
        filtered_df = filtered_df.sort_values('budget')
    elif budget:  # Fallback to categorical budget if no amount specified
        budget_mask = filtered_df['budget'].str.lower() == budget.lower()
        if not any(budget_mask):
            logger.warning(f"No destinations found with budget: {budget}")
            if filter_applied:
                return pd.DataFrame(), np.array([])
        else:
            filtered_df = filtered_df[budget_mask]
            filter_applied = True

    if filter_applied:
        # Get similarities only for filtered destinations
        filtered_indices = filtered_df.index
        filtered_similarities = similarity_series[filtered_indices]

        # Get top recommendations
        if len(filtered_similarities) == 0:
            return pd.DataFrame(), np.array([])

        top_indices = filtered_similarities.nlargest(min(top_n, len(filtered_similarities))).index
        recommendations = df.loc[top_indices]
        scores = filtered_similarities[top_indices].values
    else:
        # Get top recommendations from all destinations
        top_indices = similarity_series.nlargest(top_n).index
        recommendations = df.loc[top_indices]
        scores = similarity_series[top_indices].values

    return recommendations, scores

def load_model(model_path, num_labels):
    """Load the trained model"""
    try:
        model = DestinationRecommender(num_labels)
        # Load state dict to CPU first
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def format_recommendations(recommendations, scores):
    """
    Format recommendations in a user-friendly way.

    Args:
        recommendations (pd.DataFrame): Recommendations dataframe
        scores (np.array): Similarity scores

    Returns:
        list: List of dictionaries containing formatted recommendations
    """
    if recommendations.empty:
        return []

    formatted_recs = []
    for i, (idx, row) in enumerate(recommendations.iterrows()):
        rec = {
            'name': row['name'],
            'city': row['city'],
            'category': row['category'],
            'description': row['description'],
            'score': scores[i],
            'budget': row.get('budget', 'Not specified'),
            'operating_hours': row.get('operating hours', 'Not specified'),
            'contact_info': row.get('contact information', 'Not specified')
        }
        formatted_recs.append(rec)

    return formatted_recs

def evaluate_model(model, test_dataloader, label_encoder):
    """
    Evaluate the model on the test set and return metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # Move batch to GPU if available
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, 1)
            
            # Move predictions and labels back to CPU for metrics calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Get unique classes from both predictions and true labels
    unique_classes = np.unique(np.concatenate([all_preds, all_labels]))
    
    # Calculate metrics with zero_division=0
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Get class names for the actual classes present in the data
    class_names = [label_encoder.classes_[i] for i in unique_classes]
    
    # Generate classification report with zero_division=0
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        zero_division=0,
        labels=unique_classes
    )

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=unique_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    return {
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'classification_report': report,
        'confusion_matrix': cm
    }

def main():
    # Load data file
    file_path = "final_dataset.csv"
    model_dir = './model_output/'
    model_path = os.path.join(model_dir, 'wertigo.pt')

    # Check if model exists, if not, train a new one
    if not os.path.exists(model_path):
        print("No trained model found. Training a new model...")
        # Load and preprocess data
        df = load_data(file_path)
        df, label_encoder = preprocess_data(df)

        # Data splitting
        texts = df['combined_text'].values
        labels = df['category_encoded'].values
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, 
            labels, 
            test_size=0.3, 
            random_state=42,
        )
        # Split the temp set into validation and test sets
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=42
        )

        # Initialize tokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        # Datasets
        train_dataset = DestinationDataset(X_train, y_train, tokenizer)
        val_dataset = DestinationDataset(X_val, y_val, tokenizer)
        test_dataset = DestinationDataset(X_test, y_test, tokenizer)

        # Dataloaders
        batch_size = 4  # Small batch size due to limited data
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        # Initialize model
        num_labels = len(label_encoder.classes_)
        model = DestinationRecommender(num_labels=num_labels)
        model = model.to(device)  # Move model to GPU if available

        # Training
        model, train_losses, val_losses, val_metrics = train_model(
            model,
            train_dataloader,
            val_dataloader,
            epochs=5
        )

        # Evaluate on test set
        test_metrics = evaluate_model(model, test_dataloader, label_encoder)
        print("\nTest Set Metrics:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"F1 Score: {test_metrics['f1']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print("\nDetailed Classification Report:")
        print(test_metrics['classification_report'])

        # Save model and embeddings
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        torch.save(model.state_dict(), model_path)

        # Save embeddings
        embeddings, indices = create_destination_embeddings(model, test_dataset, test_dataloader)
        np.save(os.path.join(model_dir, 'destination_embeddings.npy'), embeddings)
        np.save(os.path.join(model_dir, 'embedding_indices.npy'), indices)

        # Save tokenizer
        tokenizer.save_pretrained(os.path.join(model_dir, 'tokenizer'))

        # Save label encoder and other metadata
        import pickle
        with open(os.path.join(model_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump({
                'label_encoder': label_encoder,
                'available_cities': df['city'].unique().tolist(),
                'available_categories': df['category'].unique().tolist(),
                'available_budgets': df['budget'].unique().tolist() if 'budget' in df.columns else None
            }, f)

        print(f"Model and resources saved to {model_dir}")
    else:
        print(f"Loading existing model from {model_path}")

    # Load all required resources
    df = load_data(file_path)
    df, label_encoder = preprocess_data(df)

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(os.path.join(model_dir, 'tokenizer')
                                               if os.path.exists(os.path.join(model_dir, 'tokenizer'))
                                               else 'roberta-base')

    # Load metadata
    import pickle
    metadata_path = os.path.join(model_dir, 'metadata.pkl')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        available_cities = metadata['available_cities']
        available_categories = metadata['available_categories']
        available_budgets = metadata['available_budgets']
    else:
        available_cities = df['city'].unique().tolist()
        available_categories = df['category'].unique().tolist()
        available_budgets = df['budget'].unique().tolist() if 'budget' in df.columns else None

    # Load model
    num_labels = len(label_encoder.classes_)
    model = load_model(model_path, num_labels)

    # Load embeddings
    embeddings_path = os.path.join(model_dir, 'destination_embeddings.npy')
    indices_path = os.path.join(model_dir, 'embedding_indices.npy')
    
    if os.path.exists(embeddings_path) and os.path.exists(indices_path):
        embeddings = np.load(embeddings_path)
        indices = np.load(indices_path)
        
        # Verify embeddings match DataFrame
        if len(embeddings) != len(df):
            print("Warning: Number of embeddings doesn't match DataFrame length. Regenerating embeddings...")
            # Create dataset for embeddings
            texts = df['combined_text'].values
            labels = df['category_encoded'].values
            test_dataset = DestinationDataset(texts, labels, tokenizer)
            test_dataloader = DataLoader(test_dataset, batch_size=4)
            embeddings, indices = create_destination_embeddings(model, test_dataset, test_dataloader)
            # Save new embeddings
            np.save(embeddings_path, embeddings)
            np.save(indices_path, indices)
    else:
        # Create dataset for embeddings
        texts = df['combined_text'].values
        labels = df['category_encoded'].values
        test_dataset = DestinationDataset(texts, labels, tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=4)
        embeddings, indices = create_destination_embeddings(model, test_dataset, test_dataloader)
        # Save embeddings
        np.save(embeddings_path, embeddings)
        np.save(indices_path, indices)

    # User interaction loop
    print("\n=== Hello, I'm Wertigo, your travel assistant! ===")
    print("Tell me what kind of place you're looking for, and I'll recommend destinations!")
    print("You can specify a city, category (e.g., historical site, natural attraction), and budget.")

    while True:
        user_query = input("\nHow can I help you? ").strip()

        if user_query.lower() == 'exit':
            print("Thank you for using the Travel Destination Recommender. Goodbye!")
            break

        if not user_query:
            print("Please enter a query or type 'exit' to quit.")
            continue

        # Extract query information
        city, category, budget, clean_query, sentiment_info, budget_amount = extract_query_info(
            user_query,
            available_cities,
            available_categories,
            available_budgets
        )

        # Error handling for missing all information. NO CATEGORY, CITY, OR BUDGET EXTRACTED!
        if not city and not category and not budget and not budget_amount:
            print("\n❌ Error: Your query is missing all required information.")
            print("Please specify at least one of the following:")
            print("- A location (e.g., 'in Tagaytay', 'in Imus')")
            print("- A category (e.g., 'cafe', 'historical site', 'natural attraction', 'resort')")
            print("- A budget (e.g., 'under 2000 pesos', 'cheap', 'luxury')")
            print("\nExample queries:")
            print("- 'I want to visit Tagaytay and I only have a budget of 500 pesos, where should I go?'")
            print("- 'Find beach resorts in Ternate under 3000 pesos'")
            print("- 'Hidden gem cafes in Silang'")
            continue

        print("\nProcessing your request...")
        print(f"Detected filters - City: {city or 'Any'}, Category: {category or 'Any'}, Budget: {budget or 'Any'}")
        if budget_amount:
            print(f"Budget amount: {budget_amount} pesos")

        # Get recommendations
        recommendations, scores = get_recommendations(
            clean_query if clean_query else user_query,
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

        # Handle the case when no recommendations are found
        if recommendations.empty:
            print("\n❌ Sorry, I couldn't find any destinations matching your criteria.")
            print("Try adjusting your search filters or use more general terms.")

            # Suggest alternatives
            suggestion = ""
            if city and category and budget:
                # Try without budget
                alt_recommendations, _ = get_recommendations(
                    clean_query if clean_query else user_query,
                    tokenizer, model, embeddings, df, city=city, category=category, budget=None, budget_amount=None, top_n=1
                )
                if not alt_recommendations.empty:
                    suggestion = f"Try searching for {category} in {city} without budget constraints."
                else:
                    # Try without category
                    alt_recommendations, _ = get_recommendations(
                        clean_query if clean_query else user_query,
                        tokenizer, model, embeddings, df, city=city, category=None, budget=None, budget_amount=None, top_n=1
                    )
                    if not alt_recommendations.empty:
                        suggestion = f"Try searching for any category in {city}."
                    else:
                        suggestion = "Try searching without any filters."
            elif city and category:
                # Try without category
                alt_recommendations, _ = get_recommendations(
                    clean_query if clean_query else user_query,
                    tokenizer, model, embeddings, df, city=city, category=None, budget=None, budget_amount=None, top_n=1
                )
                if not alt_recommendations.empty:
                    suggestion = f"Try searching for any category in {city}."
                else:
                    suggestion = "Try searching in a different city or without any filters."
            elif city:
                suggestion = "Try searching in a different city or use more general terms."
            elif category:
                suggestion = "Try searching for a different category or use more general terms."

            if suggestion:
                print(f"\nSuggestion: {suggestion}")
        else:
            # Display formatted recommendations
            print("\n✅ Here are your recommended destinations:\n")
            for i, (idx, row) in enumerate(recommendations.iterrows()):
                print(f"{i + 1}. {row['name']} ({row['city']})")
                print(f"   Category: {row['category']}")
                if 'budget' in row:
                    print(f"   Budget: ₱{row['budget']}")
                print(f"   Match score: {scores[i]:.2f}")
                print(f"   {row['description'][:150]}...\n")

if __name__ == "__main__":
    main() 