import re
import nltk
import string
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy

# Download necessary NLTK datasets
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('vader_lexicon')

# Load spaCy model for named entity recognition
nlp = spacy.load('en_core_web_sm')

# Import genres list
genres = [
    "action",
    "adventure",
    "animation",
    "biography",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "family",
    "fantasy",
    "film-noir",
    "history",
    "horror",
    "music",
    "musical",
    "mystery",
    "romance",
    "sci-fi",
    "short",
    "sport",
    "thriller",
    "war",
    "western",
    "no_genre",
]

class MovieTextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Rating words - positive and negative
        self.positive_words = [
            "excellent", "amazing", "terrific", "outstanding", "superb", 
            "brilliant", "fantastic", "wonderful", "great", "good", 
            "enjoyable", "impressive", "solid", "compelling", "masterpiece"
        ]
        
        self.negative_words = [
            "terrible", "awful", "disappointing", "poor", "bad", 
            "mediocre", "boring", "weak", "dull", "unimpressive", 
            "worst", "waste", "flawed", "horrible", "mess"
        ]
        
        # Regex patterns
        self.percentage_pattern = r'(\d{1,3})%'
        self.duration_pattern = r'(\d+)\s*(?:h|m|hr|hour|min|minute)'
        self.year_pattern = r'\b(19|20)\d{2}\b'
        self.rating_pattern = r'\b(G|PG|PG-13|R|NC-17)\b'

    def preprocess_text(self, text):
        """Basic text preprocessing"""
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return tokens
    
    def extract_movie_title(self, text):
        """Extract movie title using named entity recognition"""
        doc = nlp(text)
        # Look for WORK_OF_ART entities that might be movie titles
        for ent in doc.ents:
            if ent.label_ in ['WORK_OF_ART', 'ORG'] and len(ent.text.split()) <= 8:
                return ent.text
                
        # Alternative: look for quotes or title patterns
        title_match = re.search(r'"([^"]+)"', text) 
        if title_match:
            return title_match.group(1)
        
        # If no clear title is found
        return "Unknown Title"
    
    def extract_genres(self, text):
        """Extract movie genres from text"""
        preprocessed_text = self.preprocess_text(text)
        found_genres = []
        
        for genre in genres:
            # Convert multi-word genres to single tokens for comparison
            genre_tokens = genre.lower().replace('-', ' ').split()
            genre_lemmas = [self.lemmatizer.lemmatize(token) for token in genre_tokens]
            
            # Check if all tokens of this genre are in the text
            if all(lemma in preprocessed_text for lemma in genre_lemmas):
                found_genres.append(genre)
        
        # If no genres found, use 'no_genre' as fallback
        if not found_genres:
            found_genres.append("no_genre")
            
        return found_genres
    
    def extract_cast_crew(self, text):
        """Extract cast and crew names using NER"""
        doc = nlp(text)
        persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
        
        # Remove duplicates while preserving order
        seen = set()
        unique_persons = [x for x in persons if not (x in seen or seen.add(x))]
        
        return ", ".join(unique_persons) if unique_persons else "Not specified"
    
    def extract_metadata(self, text):
        """Extract metadata like rating, release date, duration"""
        metadata = []
        
        # Extract rating
        rating_match = re.search(self.rating_pattern, text)
        if rating_match:
            metadata.append(f"Rated {rating_match.group(1)}")
        
        # Extract year
        year_match = re.search(self.year_pattern, text)
        if year_match:
            metadata.append(f"Released {year_match.group(0)}")
        
        # Extract duration
        duration_match = re.search(self.duration_pattern, text)
        if duration_match:
            metadata.append(f"Duration {duration_match.group(0)}")
        
        return metadata
    
    def extract_score_from_percentage(self, text):
        """Extract score from percentage mentions"""
        percentage_matches = re.findall(self.percentage_pattern, text)
        if percentage_matches:
            # Return the first percentage found
            return percentage_matches[0] + "%"
        return None
    
    def calculate_sentiment_score(self, text):
        """Calculate sentiment score as a percentage"""
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        # Convert sentiment compound score to a 0-100% scale
        score = int((sentiment['compound'] + 1) * 50)
        return f"{score}%"
    
    def extract_critics_score(self, text):
        """Extract critics score from text"""
        # First check for direct percentage mention
        score = self.extract_score_from_percentage(text)
        if score:
            return score
        
        # If no direct percentage, calculate from sentiment
        return self.calculate_sentiment_score(text)
    
    def extract_audience_score(self, text):
        """Extract audience score from text"""
        # First check for direct percentage mention
        score = self.extract_score_from_percentage(text)
        if score:
            return score
        
        # If no direct percentage, calculate from sentiment
        return self.calculate_sentiment_score(text)
    
    def extract_verified_count(self, text):
        """Extract audience verified count if mentioned"""
        count_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s+(?:verified|ratings|reviews)', text, re.IGNORECASE)
        if count_match:
            return count_match.group(1)
        return "Not specified"
    
    def extract_summary(self, text, is_critics=True):
        """Extract a summary from text"""
        sentences = sent_tokenize(text)
        if not sentences:
            return "No summary available"
        
        # Get sentiment for each sentence
        sentiments = [self.sentiment_analyzer.polarity_scores(sent)['compound'] for sent in sentences]
        
        # For critics, we might want sentences with stronger sentiments (positive or negative)
        # For audience, we might want more balanced views
        if is_critics:
            # Get sentence with strongest sentiment (either direction)
            idx = np.argmax(np.abs(sentiments))
        else:
            # Get a sentence with moderate sentiment
            idx = np.argmin(np.abs(np.array(sentiments) - np.mean(sentiments)))
        
        return sentences[idx]
    
    def process_movie_data(self, text):
        """Process movie description text to extract all required information"""
        result = {
            'movieTitle': self.extract_movie_title(text),
            'genres': self.extract_genres(text),
            'movieDesc': text[:200] + "..." if len(text) > 200 else text,  # Truncate long descriptions
            'cast': self.extract_cast_crew(text),
            'metadataArr': self.extract_metadata(text),
            'criticsScore': self.extract_critics_score(text),
            'audienceScore': self.extract_audience_score(text),
            'audienceVerifiedCount': self.extract_verified_count(text),
            'criticsConsensus': self.extract_summary(text, is_critics=True),
            'audienceConsensus': self.extract_summary(text, is_critics=False)
        }
        
        # For critic reviews, we'd need separate text input
        result['criticReviews'] = "Not provided"
        
        return result
    
    def format_for_embedding(self, data):
        """Format the extracted data for embedding according to the specified format"""
        text_parts = [
            f"The movie title is {data['movieTitle']}",
            f"The movie genres are {', '.join(data['genres'])}",
            f"The movie description is {data['movieDesc']}",
            f"The movie cast and crews are {data['cast']}",
            f"Metadata(rating, released Date, duration): {', '.join(data['metadataArr'])}" if data['metadataArr'] else None,
            f"Critics Score: {data['criticsScore']}",
            f"Critics Review: {data['criticReviews']}",
            f"Audience Score: {data['audienceScore']}",
            f"Audience verified count: {data['audienceVerifiedCount']}",
            f"Critics summary regard: {data['criticsConsensus']}",
            f"Audience summary regard: {data['audienceConsensus']}",
        ]
        
        # Filter out None values and join with semicolons
        formatted_text = "; ".join([part for part in text_parts if part])
        return formatted_text


# Example usage
if __name__ == "__main__":
    processor = MovieTextProcessor()
    
    sample_text = """
    Recommend me some thriller, sci-fi movies, high rated in critics and audience.
    """
    
    # Process the data
    movie_data = processor.process_movie_data(sample_text)
    
    # Format for embedding
    embedding_text = processor.format_for_embedding(movie_data)
    
    print("Extracted Movie Data:")
    for key, value in movie_data.items():
        print(f"{key}: {value}")
    
    print("\nFormatted for Embedding:")
    print(embedding_text)