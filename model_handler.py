import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import importlib.util
import logging
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the utils directory to the path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# Constants
MODEL_OUTPUT_DIR = os.path.join(CURRENT_DIR, 'model_output')
HF_REPO = "chanel999/wertigo"  # Match the repository name from upload_model.py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_FILENAME = "wertigo.pt"

# Global variables
model = None
df = None
embeddings = None
tokenizer = None

# Initialize model state
logger.info(f"Using device: {device}")

# Import from revised.py
def import_model_components():
    try:
        # Use importlib.util to load the revised module
        revised_path = os.path.join(CURRENT_DIR, "revised.py")
        spec = importlib.util.spec_from_file_location("travel_model", revised_path)
        travel_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(travel_model)
        
        # Import the required components from revised.py
        return travel_model.DestinationRecommender, travel_model.extract_query_info, travel_model.load_data, travel_model.preprocess_data
    except Exception as e:
        logger.error(f"Error importing from revised.py: {e}")
        raise

# Load model from saved state
def load_model(model_path=None):
    """
    Load the model from Hugging Face or local path
    """
    try:
        # Try to download from Hugging Face first
        logger.info(f"Attempting to download model from {HF_REPO}")
        model_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=MODEL_FILENAME
        )
        logger.info(f"Successfully downloaded model to {model_path}")
    except Exception as e:
        logger.warning(f"Failed to download from Hugging Face: {e}")
        if model_path is None:
            model_path = os.path.join("model_output", MODEL_FILENAME)
        logger.info(f"Using local model at {model_path}")
    
    # Ensure model_output directory exists
    os.makedirs("model_output", exist_ok=True)
    
    # Load the model
    try:
        model = torch.load(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def save_model(model, model_path=None):
    """
    Save the model to a file
    """
    if model_path is None:
        model_path = os.path.join("model_output", MODEL_FILENAME)
    
    # Ensure model_output directory exists
    os.makedirs("model_output", exist_ok=True)
    
    try:
        torch.save(model, model_path)
        logger.info(f"Model saved successfully to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def create_model(input_size, hidden_size, output_size):
    """
    Create a new model instance
    """
    return PokemonModel(input_size, hidden_size, output_size)

class PokemonModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PokemonModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load embeddings from file or create them
def get_embeddings(model, df):
    """
    Load or create embeddings for destinations
    """
    embeddings_path = os.path.join(MODEL_OUTPUT_DIR, 'destination_embeddings.npy')
    
    try:
        if os.path.exists(embeddings_path):
            loaded_embeddings = np.load(embeddings_path)
            # Check if the shape matches our current data
            if len(loaded_embeddings) == len(df):
                logger.info(f"Loaded embeddings for {len(loaded_embeddings)} destinations.")
                return loaded_embeddings
            else:
                logger.warning(f"Embeddings size mismatch. Expected {len(df)}, got {len(loaded_embeddings)}. Creating new embeddings...")
        else:
            logger.info("Embeddings file not found. Creating new embeddings...")
        
        # If embeddings don't exist or don't match, create them
        logger.info(f"Generating embeddings for {len(df)} destinations...")
        embeddings = []
        
        # Process in batches to avoid memory issues
        batch_size = 8  # Smaller batch size for safety
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            if i % 32 == 0:  # Log every 32 items
                logger.info(f"Processing destinations {i}/{len(df)}...")
            
            with torch.no_grad():
                for _, row in batch_df.iterrows():
                    # Use the combined_text field
                    text = str(row['combined_text'])
                    
                    # Truncate if too long
                    if len(text) > 1000:
                        text = text[:1000]
                    
                    encoding = tokenizer(
                        text,
                        add_special_tokens=True,
                        max_length=512,
                        return_token_type_ids=False,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                    ).to(device)
                    
                    outputs = model.roberta(
                        input_ids=encoding['input_ids'],
                        attention_mask=encoding['attention_mask']
                    )
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(embedding[0])  # Remove the batch dimension
        
        # Convert to numpy array
        embeddings = np.array(embeddings)
        
        # Verify the embeddings shape
        if len(embeddings) != len(df):
            logger.error(f"Generated embeddings count ({len(embeddings)}) doesn't match destinations count ({len(df)})")
            return np.array([])
        
        # Save the embeddings
        try:
            os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
            np.save(embeddings_path, embeddings)
            logger.info(f"Saved embeddings for {len(embeddings)} destinations.")
        except Exception as e:
            logger.warning(f"Failed to save embeddings: {e}")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])

# Initialize the model at module import time
def init_model():
    """
    Initialize the model, data, and embeddings
    """
    global model, df, embeddings, tokenizer
    
    try:
        logger.info("Loading recommendation model...")
        model, df, label_encoder = load_model()
        
        if model is not None and df is not None:
            # Initialize tokenizer from transformers
            from transformers import RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            
            logger.info("Loading embeddings...")
            embeddings = get_embeddings(model, df)
            if embeddings is None or len(embeddings) == 0:
                logger.warning("No embeddings available. Some features may not work correctly.")
                logger.info("You may need to train the model first by running: python revised.py")
        else:
            logger.warning("Model or data not available. Recommendation features will be limited.")
            logger.info("Please ensure final_dataset.csv exists and run: python revised.py to train the model")
            
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        logger.error("Starting with limited functionality. Recommendation features will not be available.")

# Initialize the model when this module is imported
init_model() 