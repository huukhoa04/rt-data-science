import re
import nltk
import string
import numpy as np
import datetime
from collections import defaultdict
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

# Updated genres list based on text-processor.md
genres = [
    "action",
    "adventure",
    "animation",
    "anime",
    "biography",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "entertainment",
    "fantasy",
    "health & wellness",
    "history",
    "holiday",
    "horror",
    "house & garden",
    "kids & family",
    "music",
    "musical",
    "mystery & thriller",
    "nature",
    "news",
    "reality",
    "romance",
    "sci-fi",
    "short",
    "soap",
    "special interest",
    "sports",
    "stand-up",
    "talk show",
    "travel",
    "variety",
    "war",
    "western"
]

class MovieTextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Add more stopwords that don't change meaning
        additional_stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'being', 'been', 
                              'have', 'has', 'had', 'do', 'does', 'did', 'could', 'should', 'would',
                              'can', 'will', 'shall', 'may', 'might', 'must', 'am'}
        self.stop_words.update(additional_stopwords)
        
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Updated critic word lists with standardization mappings
        self.critics_positive = [
            "exceptional", "masterpiece", "brilliant", "outstanding", "superb", 
            "certified fresh", "excellent", "very good", "well-crafted", "impressive",
            "fresh", "good", "solid", "enjoyable", "decent"
        ]
        
        self.critics_negative = [
            "rotten", "mediocre", "underwhelming", "flawed", "unremarkable",
            "extremely rotten", "poor", "disappointing", "subpar", "problematic",
            "awful", "terrible", "abysmal", "unwatchable", "complete failure"
        ]
        
        # Critics word standardization mappings
        self.critics_positive_mapping = {
            # Exceptional equivalents
            "perfect": "exceptional", "flawless": "exceptional", "magnificent": "exceptional",
            "extraordinary": "exceptional", "remarkable": "exceptional", "sublime": "exceptional",
            
            # Masterpiece equivalents
            "classic": "masterpiece", "tour de force": "masterpiece", "triumph": "masterpiece",
            "chef-d'oeuvre": "masterpiece", "crowning achievement": "masterpiece",
            
            # Brilliant equivalents
            "genius": "brilliant", "ingenious": "brilliant", "dazzling": "brilliant",
            "stellar": "brilliant", "spectacular": "brilliant", "amazing": "brilliant",
            
            # Outstanding equivalents
            "standout": "outstanding", "excellent": "outstanding", "exceptional": "outstanding",
            "phenomenal": "outstanding", "terrific": "outstanding", "fantastic": "outstanding",
            
            # Superb equivalents
            "top-notch": "superb", "first-rate": "superb", "marvelous": "superb",
            "wonderful": "superb", "exquisite": "superb", "splendid": "superb",
            
            # Certified Fresh equivalents
            "fresh": "certified fresh", "acclaimed": "certified fresh", "critically acclaimed": "certified fresh",
            
            # Excellent equivalents
            "superior": "excellent", "extraordinary": "excellent", "great": "excellent",
            "exceptional": "excellent", "outstanding": "excellent",
            
            # Very Good equivalents
            "well done": "very good", "admirable": "very good", "commendable": "very good",
            "praiseworthy": "very good", "noteworthy": "very good",
            
            # Well-Crafted equivalents
            "well-made": "well-crafted", "skillful": "well-crafted", "polished": "well-crafted",
            "refined": "well-crafted", "meticulous": "well-crafted",
            
            # Impressive equivalents
            "striking": "impressive", "remarkable": "impressive", "notable": "impressive",
            "memorable": "impressive", "powerful": "impressive",
            
            # Fresh equivalents (Rotten Tomatoes specific)
            "worthwhile": "fresh", "recommended": "fresh", "above average": "fresh",
            
            # Good equivalents
            "satisfying": "good", "pleasing": "good", "agreeable": "good",
            "fine": "good", "nice": "good", "gratifying": "good",
            
            # Solid equivalents
            "sturdy": "solid", "reliable": "solid", "dependable": "solid",
            "consistent": "solid", "steady": "solid", "sound": "solid",
            
            # Enjoyable equivalents
            "entertaining": "enjoyable", "fun": "enjoyable", "delightful": "enjoyable",
            "pleasant": "enjoyable", "amusing": "enjoyable", "engaging": "enjoyable",
            
            # Decent equivalents
            "adequate": "decent", "acceptable": "decent", "fair": "decent",
            "reasonable": "decent", "passable": "decent", "sufficient": "decent",
        }
        
        self.critics_negative_mapping = {
            # Rotten equivalents (Rotten Tomatoes specific)
            "bad": "rotten", "unpleasant": "rotten", "disagreeable": "rotten", 
            "unfavorable": "rotten", "negative": "rotten",
            
            # Mediocre equivalents
            "average": "mediocre", "ordinary": "mediocre", "so-so": "mediocre",
            "middling": "mediocre", "uninspired": "mediocre", "bland": "mediocre",
            
            # Underwhelming equivalents
            "unimpressive": "underwhelming", "lackluster": "underwhelming", "unexciting": "underwhelming",
            "tepid": "underwhelming", "tame": "underwhelming", "dull": "underwhelming",
            
            # Flawed equivalents
            "imperfect": "flawed", "defective": "flawed", "deficient": "flawed",
            "faulty": "flawed", "weak": "flawed", "problematic": "flawed",
            
            # Unremarkable equivalents
            "forgettable": "unremarkable", "unmemorable": "unremarkable", "unexceptional": "unremarkable",
            "nondescript": "unremarkable", "commonplace": "unremarkable", "insignificant": "unremarkable",
            
            # Extremely Rotten equivalents
            "dreadful": "extremely rotten", "atrocious": "extremely rotten", "horrendous": "extremely rotten",
            "appalling": "extremely rotten", "deplorable": "extremely rotten",
            
            # Poor equivalents
            "inferior": "poor", "substandard": "poor", "unsatisfactory": "poor",
            "inadequate": "poor", "insufficient": "poor", "lacking": "poor",
            
            # Disappointing equivalents
            "letdown": "disappointing", "dissatisfying": "disappointing", "unfulfilling": "disappointing",
            "frustrating": "disappointing", "disheartening": "disappointing",
            
            # Subpar equivalents
            "below average": "subpar", "below par": "subpar", "second-rate": "subpar",
            "third-rate": "subpar", "inferior": "subpar",
            
            # Problematic equivalents
            "troublesome": "problematic", "difficult": "problematic", "challenging": "problematic",
            "complicated": "problematic", "questionable": "problematic",
            
            # Awful equivalents
            "horrible": "awful", "terrible": "awful", "dreadful": "awful",
            "appalling": "awful", "ghastly": "awful", "dire": "awful",
            
            # Terrible equivalents
            "horrific": "terrible", "horrendous": "terrible", "atrocious": "terrible",
            "abysmal": "terrible", "abhorrent": "terrible", "detestable": "terrible",
            
            # Abysmal equivalents
            "rock-bottom": "abysmal", "worthless": "abysmal", "dismal": "abysmal",
            "disastrous": "abysmal", "catastrophic": "abysmal",
            
            # Unwatchable equivalents
            "intolerable": "unwatchable", "unbearable": "unwatchable", "insufferable": "unwatchable",
            "excruciating": "unwatchable", "torturous": "unwatchable",
            
            # Complete Failure equivalents
            "fiasco": "complete failure", "debacle": "complete failure", "disaster": "complete failure",
            "train wreck": "complete failure", "catastrophe": "complete failure", "mess": "complete failure",
        }
        
        # Updated audience word lists with standardization mappings
        self.audience_positive = [
            "must see", "incredible", "loved every minute", "absolutely amazing", "life-changing",
            "loved it", "thoroughly enjoyed", "great experience", "highly recommend", "worth watching",
            "liked it", "pretty good", "entertained", "satisfying", "worthwhile"
        ]
        
        self.audience_negative = [
            "mixed feelings", "so-so", "has issues", "avoid it", "waste of time",
            "regrettable", "painful experience", "unbearable", "average at best", "nothing special",
            "disliked it", "not worth it", "skip this one", "letdown", "frustrating"
        ]
        
        # Audience word standardization mappings
        self.audience_positive_mapping = {
            # Must See equivalents
            "don't miss": "must see", "essential viewing": "must see", "can't miss": "must see",
            "mandatory": "must see", "required viewing": "must see",
            
            # Incredible equivalents
            "amazing": "incredible", "fantastic": "incredible", "astonishing": "incredible",
            "unbelievable": "incredible", "mind-blowing": "incredible", "astounding": "incredible",
            
            # Loved Every Minute equivalents
            "never a dull moment": "loved every minute", "captivating throughout": "loved every minute",
            "enthralled": "loved every minute", "gripping": "loved every minute",
            
            # Absolutely Amazing equivalents
            "breathtaking": "absolutely amazing", "jaw-dropping": "absolutely amazing",
            "phenomenal": "absolutely amazing", "exceptional": "absolutely amazing",
            
            # Life-Changing equivalents
            "transformative": "life-changing", "profound": "life-changing",
            "inspirational": "life-changing", "eye-opening": "life-changing",
            
            # Loved It equivalents
            "adored": "loved it", "enjoyed immensely": "loved it", "fantastic": "loved it",
            "great": "loved it", "wonderful": "loved it",
            
            # Thoroughly Enjoyed equivalents
            "delighted by": "thoroughly enjoyed", "completely enjoyed": "thoroughly enjoyed",
            "fully engaged": "thoroughly enjoyed", "immersed in": "thoroughly enjoyed",
            
            # Great Experience equivalents
            "wonderful time": "great experience", "excellent experience": "great experience",
            "memorable experience": "great experience", "positive experience": "great experience",
            
            # Highly Recommend equivalents
            "strongly recommend": "highly recommend", "enthusiastically recommend": "highly recommend",
            "wholeheartedly recommend": "highly recommend", "definitely recommend": "highly recommend",
            
            # Worth Watching equivalents
            "deserves viewing": "worth watching", "deserves your time": "worth watching",
            "time well spent": "worth watching", "worthy of attention": "worth watching",
            
            # Liked It equivalents
            "enjoyed": "liked it", "pleased with": "liked it", "favorable": "liked it",
            "positive": "liked it", "good": "liked it",
            
            # Pretty Good equivalents
            "better than expected": "pretty good", "quite good": "pretty good",
            "relatively good": "pretty good", "rather enjoyable": "pretty good",
            
            # Entertained equivalents
            "amused": "entertained", "diverted": "entertained", "engaged": "entertained",
            "interested": "entertained", "absorbed": "entertained",
            
            # Satisfying equivalents
            "fulfilling": "satisfying", "gratifying": "satisfying", "pleasing": "satisfying",
            "rewarding": "satisfying", "adequate": "satisfying",
            
            # Worthwhile equivalents
            "valuable": "worthwhile", "meaningful": "worthwhile", "justified": "worthwhile",
            "merited": "worthwhile", "deserving": "worthwhile",
        }
        
        self.audience_negative_mapping = {
            # Mixed Feelings equivalents
            "ambivalent": "mixed feelings", "conflicted": "mixed feelings", "torn": "mixed feelings",
            "uncertain": "mixed feelings", "undecided": "mixed feelings",
            
            # So-So equivalents
            "mediocre": "so-so", "average": "so-so", "middling": "so-so",
            "fair": "so-so", "passable": "so-so", "okay": "so-so",
            
            # Has Issues equivalents
            "problematic": "has issues", "flawed": "has issues", "shortcomings": "has issues",
            "deficiencies": "has issues", "faults": "has issues",
            
            # Avoid It equivalents
            "stay away": "avoid it", "skip it": "avoid it", "pass on this": "avoid it",
            "give it a miss": "avoid it", "don't bother": "avoid it",
            
            # Waste of Time equivalents
            "time waster": "waste of time", "not worth the time": "waste of time",
            "squandered hours": "waste of time", "time lost": "waste of time",
            
            # Regrettable equivalents
            "unfortunate": "regrettable", "disappointment": "regrettable", "mistake": "regrettable",
            "wish I hadn't": "regrettable", "poor choice": "regrettable",
            
            # Painful Experience equivalents
            "torturous": "painful experience", "agonizing": "painful experience",
            "excruciating": "painful experience", "distressing": "painful experience",
            
            # Unbearable equivalents
            "intolerable": "unbearable", "insufferable": "unbearable", "unendurable": "unbearable",
            "too much": "unbearable", "couldn't finish": "unbearable",
            
            # Average at Best equivalents
            "nothing special": "average at best", "unremarkable": "average at best",
            "ordinary": "average at best", "run-of-the-mill": "average at best",
            
            # Nothing Special equivalents
            "forgettable": "nothing special", "bland": "nothing special", "generic": "nothing special",
            "dull": "nothing special", "uninspired": "nothing special",
            
            # Disliked It equivalents
            "didn't enjoy": "disliked it", "wasn't a fan": "disliked it", "not for me": "disliked it",
            "didn't care for": "disliked it", "unpleasant": "disliked it",
            
            # Not Worth It equivalents
            "not worth the money": "not worth it", "not worth the effort": "not worth it",
            "waste": "not worth it", "overpriced": "not worth it",
            
            # Skip This One equivalents
            "don't watch": "skip this one", "ignore this one": "skip this one",
            "pass": "skip this one", "next": "skip this one",
            
            # Letdown equivalents
            "disappointing": "letdown", "fell short": "letdown", "didn't meet expectations": "letdown",
            "underwhelming": "letdown", "not as good as expected": "letdown",
            
            # Frustrating equivalents
            "annoying": "frustrating", "irritating": "frustrating", "exasperating": "frustrating",
            "infuriating": "frustrating", "vexing": "frustrating",
        }
        
        # Regex patterns
        self.percentage_pattern = r'(\d{1,3})%'
        self.duration_pattern = r'(\d+)\s*(?:h|m|hr|hour|min|minute)'
        self.year_pattern = r'\b(19|20)\d{2}\b'
        self.rating_pattern = r'\b(G|PG|PG-13|R|NC-17)\b'
        self.month_pattern = r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b'
        self.runtime_pattern = r'\b(\d+)h\s*(\d+)m|\b(\d+)\s*(?:hour|hr)s?\s*(\d+)\s*(?:min|minute)s?|\b(\d+)\s*(?:hour|hr)s?|\b(\d+)\s*(?:min|minute)s?'

    def preprocess_text(self, text):
        """Enhanced text preprocessing with better lemmatization"""
        if not text:
            return []
            
        text = text.lower()
        # Preserve contractions like don't, won't
        text = re.sub(r'[^\w\s\']', ' ', text)
        tokens = word_tokenize(text)
        
        # More thorough lemmatization with POS tagging for better results
        pos_tagged = pos_tag(tokens)
        lemmatized_tokens = []
        
        for word, tag in pos_tagged:
            if word not in self.stop_words:
                # Get appropriate POS for lemmatization
                if tag.startswith('J'):
                    pos = 'a'  # adjective
                elif tag.startswith('V'):
                    pos = 'v'  # verb
                elif tag.startswith('N'):
                    pos = 'n'  # noun
                elif tag.startswith('R'):
                    pos = 'r'  # adverb
                else:
                    pos = 'n'  # default to noun
                
                lemmatized_tokens.append(self.lemmatizer.lemmatize(word, pos))
                
        return lemmatized_tokens
    
    def standardize_critics_terms(self, text):
        """Standardize critic terms to the preferred vocabulary"""
        if not text:
            return text
            
        result = text.lower()
        
        # Replace positive terms with standard terms
        for term, standard in self.critics_positive_mapping.items():
            result = re.sub(r'\b' + re.escape(term) + r'\b', standard, result)
            
        # Replace negative terms with standard terms
        for term, standard in self.critics_negative_mapping.items():
            result = re.sub(r'\b' + re.escape(term) + r'\b', standard, result)
            
        return result
    
    def standardize_audience_terms(self, text):
        """Standardize audience terms to the preferred vocabulary"""
        if not text:
            return text
            
        result = text.lower()
        
        # Replace positive terms with standard terms
        for term, standard in self.audience_positive_mapping.items():
            result = re.sub(r'\b' + re.escape(term) + r'\b', standard, result)
            
        # Replace negative terms with standard terms
        for term, standard in self.audience_negative_mapping.items():
            result = re.sub(r'\b' + re.escape(term) + r'\b', standard, result)
            
        return result
    
    def is_movie_query(self, text):
        """Determine if the input text is a movie query or description"""
        query_indicators = ['recommend', 'suggest', 'looking for', 'find', 'search', 
                           'movie like', 'similar to', 'what are some', 'can you suggest']
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Check if any query indicators are present
        for indicator in query_indicators:
            if indicator in text_lower:
                return True
                
        return False
    
    def extract_movie_title(self, text):
        """Extract movie title using named entity recognition"""
        if not text:
            return "Unknown Title"
            
        # Check if this is a movie query rather than a description
        if self.is_movie_query(text):
            return "Movie Query"
            
        doc = nlp(text)
        
        # Look for WORK_OF_ART entities that might be movie titles
        for ent in doc.ents:
            if ent.label_ in ['WORK_OF_ART', 'ORG', 'EVENT'] and len(ent.text.split()) <= 8:
                return ent.text
                
        # Search for quoted text which often contains titles
        title_matches = re.findall(r'"([^"]+)"', text)
        if title_matches:
            return title_matches[0]
            
        title_matches = re.findall(r'\'([^\']+)\'', text)
        if title_matches:
            return title_matches[0]
            
        # Look for text that follows "titled" or "called"
        titled_match = re.search(r'(?:titled|called|named)\s+["\']?([^,"\'\.]+)["\']?', text, re.IGNORECASE)
        if titled_match:
            return titled_match.group(1).strip()
        
        # If no clear title is found
        return "Unknown Title"
    
    def extract_genres(self, text):
        """Extract movie genres from text with improved accuracy"""
        if not text:
            return ["no_genre"]
            
        preprocessed_text = " ".join(self.preprocess_text(text))
        found_genres = []
        
        for genre in genres:
            # Handle multi-word genres and different formats
            search_genre = genre.lower().replace('&', 'and').replace('-', ' ')
            
            # Special case for "mystery & thriller"
            if genre == "mystery & thriller":
                if "mystery" in preprocessed_text or "thriller" in preprocessed_text:
                    found_genres.append(genre)
                continue
                
            # Check for direct mention or variations
            if search_genre in preprocessed_text or genre in preprocessed_text:
                found_genres.append(genre)
                continue
                
            # For compound genres, check if all parts are present
            parts = search_genre.split()
            if len(parts) > 1 and all(part in preprocessed_text.split() for part in parts):
                found_genres.append(genre)
                
        # Check for query type that might be looking for genres
        if self.is_movie_query(text):
            genre_query_terms = ["action", "adventure", "comedy", "horror", "thriller", 
                                "drama", "sci-fi", "fantasy", "romance", "mystery"]
            
            for term in genre_query_terms:
                if term in preprocessed_text:
                    found_genres.append(term)
        
        # If no genres found, use 'no_genre' as fallback
        if not found_genres:
            found_genres.append("no_genre")
            
        return found_genres
    
    def extract_cast_crew(self, text):
        """Extract cast and crew names using improved NER"""
        if not text:
            return "Not specified"
            
        doc = nlp(text)
        
        # Extract all person entities
        persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
        
        # Find names preceded by role indicators
        role_patterns = [
            r'(?:starring|stars|featuring|features|with|directed by|director|produced by|producer|written by|writer)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:plays|portrayed|acted)'
        ]
        
        for pattern in role_patterns:
            role_matches = re.findall(pattern, text)
            persons.extend(role_matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_persons = [x for x in persons if not (x.lower() in seen or seen.add(x.lower()))]
        
        if unique_persons:
            return ", ".join(unique_persons)
        else:
            return "Not specified"
    
    def format_date(self, date_text):
        """Format release date to 'Released Mon DD' format"""
        if not date_text:
            return None
            
        # Extract month and day
        month_day_match = re.search(self.month_pattern, date_text)
        if month_day_match:
            return f"Released {month_day_match.group(0)}"
            
        # Extract year
        year_match = re.search(self.year_pattern, date_text)
        if year_match:
            return f"Released {year_match.group(0)}"
            
        return None
    
    def format_runtime(self, runtime_text):
        """Format runtime to '1h 40m' format"""
        if not runtime_text:
            return None
            
        match = re.search(self.runtime_pattern, runtime_text)
        if not match:
            return None
            
        groups = match.groups()
        
        # Format: "1h 30m"
        if groups[0] and groups[1]:
            return f"{groups[0]}h {groups[1]}m"
            
        # Format: "1 hour 30 minutes"
        elif groups[2] and groups[3]:
            return f"{groups[2]}h {groups[3]}m"
            
        # Format: "1 hour" only
        elif groups[4]:
            return f"{groups[4]}h"
            
        # Format: "90 minutes" only
        elif groups[5]:
            return f"{groups[5]}m"
            
        return None
    
    def extract_metadata(self, text):
        """Extract and format metadata like rating, release date, duration"""
        if not text:
            return []
            
        metadata = []
        
        # Extract rating
        rating_match = re.search(self.rating_pattern, text)
        if rating_match:
            metadata.append(f"{rating_match.group(1)}")
        
        # Extract and format release date
        date_format = self.format_date(text)
        if date_format:
            metadata.append(date_format)
        
        # Extract and format runtime
        runtime_format = self.format_runtime(text)
        if runtime_format:
            metadata.append(runtime_format)
        
        return metadata
    
    def extract_score_from_percentage(self, text):
        """Extract score from percentage mentions"""
        if not text:
            return None
            
        percentage_matches = re.findall(self.percentage_pattern, text)
        if percentage_matches:
            # Return the first percentage found
            return percentage_matches[0] + "%"
        return None
    
    def calculate_sentiment_score(self, text, is_critic=True):
        """Calculate sentiment score as a percentage with improved accuracy"""
        if not text:
            return "0%"
            
        # Standardize terms before sentiment analysis
        if is_critic:
            standardized_text = self.standardize_critics_terms(text)
        else:
            standardized_text = self.standardize_audience_terms(text)
            
        # Count standard positive and negative terms
        pos_count = 0
        neg_count = 0
        
        words = word_tokenize(standardized_text.lower())
        
        if is_critic:
            for word in words:
                if word in self.critics_positive:
                    pos_count += 1
                elif word in self.critics_negative:
                    neg_count += 1
        else:
            for word in words:
                if word in self.audience_positive:
                    pos_count += 1
                elif word in self.audience_negative:
                    neg_count += 1
        
        # Use VADER as a fallback if no standard terms found
        if pos_count == 0 and neg_count == 0:
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            # Convert compound score to a 0-100% scale
            score = int((sentiment['compound'] + 1) * 50)
            return f"{score}%"
        
        # Calculate percentage based on the ratio of positive to total opinion words
        total = pos_count + neg_count
        if total > 0:
            score = int((pos_count / total) * 100)
            return f"{score}%"
        
        # Default to neutral if no sentiment information
        return "50%"
    
    def extract_critics_sentiment(self, text):
        """Extract critics sentiment using the standardized wordset"""
        if not text:
            return "Good"  # Default neutral sentiment
            
        # First check for direct mentions of the standardized terms
        lower_text = text.lower()
        
        # Check for exact matches of the specified terms first
        for term in self.critics_positive:
            if term.lower() in lower_text:
                return term.capitalize()
                
        for term in self.critics_negative:
            if term.lower() in lower_text:
                return term.capitalize()
        
        # If no direct match, standardize the text and look for standardized terms
        standardized_text = self.standardize_critics_terms(text)
        words = word_tokenize(standardized_text.lower())
        
        # Check for standardized terms after conversion
        for word in words:
            if word in self.critics_positive:
                return word.capitalize()
            if word in self.critics_negative:
                return word.capitalize()
        
        # If still no match, use VADER to select an appropriate term
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        compound = sentiment['compound']
        
        # Map sentiment score to appropriate term from the specified list
        if compound >= 0.6:
            return "Outstanding"
        elif compound >= 0.4:
            return "Excellent"
        elif compound >= 0.2:
            return "Good"
        elif compound >= 0.0:
            return "Decent"
        elif compound >= -0.2:
            return "Mediocre"
        elif compound >= -0.4:
            return "Disappointing"
        elif compound >= -0.6:
            return "Poor"
        else:
            return "Terrible"
    
    def extract_audience_sentiment(self, text):
        """Extract audience sentiment using the standardized wordset"""
        if not text:
            return "Liked It"  # Default neutral sentiment
            
        # First check for direct mentions of the standardized terms
        lower_text = text.lower()
        
        # Check for exact matches of multi-word terms first (to handle phrases correctly)
        for term in self.audience_positive:
            if term.lower() in lower_text:
                return term.capitalize()
                
        for term in self.audience_negative:
            if term.lower() in lower_text:
                return term.capitalize()
        
        # If no direct match, standardize the text and check again
        standardized_text = self.standardize_audience_terms(text)
        
        # For audience terms that may be phrases, we need to check the full text
        for term in self.audience_positive:
            if term.lower() in standardized_text.lower():
                return term.capitalize()
                
        for term in self.audience_negative:
            if term.lower() in standardized_text.lower():
                return term.capitalize()
        
        # If still no match, use VADER to select an appropriate term
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        compound = sentiment['compound']
        
        # Map sentiment score to appropriate term from the specified list
        if compound >= 0.6:
            return "Loved It"
        elif compound >= 0.4:
            return "Worth Watching"
        elif compound >= 0.2:
            return "Liked It"
        elif compound >= 0.0:
            return "Pretty Good"
        elif compound >= -0.2:
            return "So-So"
        elif compound >= -0.4:
            return "Has Issues"
        elif compound >= -0.6:
            return "Not Worth It"
        else:
            return "Avoid It"
    
    def extract_verified_count(self, text):
        """Extract audience verified count with improved pattern matching"""
        if not text:
            return "Not specified"
            
        # Match patterns like "5,000 verified", "1,234 ratings", etc.
        count_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s+(?:verified|ratings|reviews|votes|audience)', text, re.IGNORECASE)
        if count_match:
            return count_match.group(1)
            
        # Match patterns like "verified by 5,000 users"
        alt_match = re.search(r'(?:verified|rated|reviewed)\s+by\s+(\d{1,3}(?:,\d{3})*)', text, re.IGNORECASE)
        if alt_match:
            return alt_match.group(1)
            
        return "Not specified"
    
    def extract_summary(self, text, is_critics=True):
        """Extract a more relevant summary from text"""
        if not text:
            return "No summary available"
            
        # Standardize terms based on the audience
        if is_critics:
            standardized_text = self.standardize_critics_terms(text)
        else:
            standardized_text = self.standardize_audience_terms(text)
            
        sentences = sent_tokenize(standardized_text)
        if not sentences:
            return "No summary available"
        
        # Score sentences based on sentiment and relevant terms
        sentence_scores = []
        
        for sent in sentences:
            sentiment = self.sentiment_analyzer.polarity_scores(sent)
            score = abs(sentiment['compound'])  # We want strong sentiment (positive or negative)
            
            # Boost score for sentences with standard terms
            lower_sent = sent.lower()
            if is_critics:
                for term in self.critics_positive + self.critics_negative:
                    if term in lower_sent:
                        score += 0.2
            else:
                for term in self.audience_positive + self.audience_negative:
                    if term in lower_sent:
                        score += 0.2
            
            # Boost score for sentences with movie-related content
            movie_terms = ['film', 'movie', 'story', 'plot', 'character', 'acting', 'director', 'scene', 'performance']
            for term in movie_terms:
                if term in lower_sent:
                    score += 0.1
            
            sentence_scores.append((sent, score))
        
        # Sort by score and get the highest-scoring sentence
        sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        
        # Get the highest scoring sentence that's not too short or too long
        for sent, score in sorted_sentences:
            words = sent.split()
            if 5 <= len(words) <= 25:
                return sent
        
        # If no ideal sentence found, return the highest scoring one
        if sorted_sentences:
            return sorted_sentences[0][0]
        
        return "No informative summary available"
    
    def extract_movie_description(self, text):
        """Extract a concise movie description from text"""
        if not text:
            return "No description available"
            
        # Check if this is a query
        if self.is_movie_query(text):
            return text  # Return the query as is
            
        # Extract sentences that describe the movie plot
        plot_indicators = ['about', 'follows', 'tells', 'story', 'plot', 'narrative', 
                         'centers on', 'focuses on', 'features', 'portrays', 'depicts']
        
        sentences = sent_tokenize(text)
        plot_sentences = []
        
        for sent in sentences:
            lower_sent = sent.lower()
            if any(indicator in lower_sent for indicator in plot_indicators):
                plot_sentences.append(sent)
        
        # If we found plot sentences, join them (up to 2)
        if plot_sentences:
            return " ".join(plot_sentences[:2])
        
        # Otherwise, take the first 1-2 sentences that aren't too short
        eligible_sentences = [s for s in sentences if len(s.split()) > 5]
        if eligible_sentences:
            return " ".join(eligible_sentences[:2])
        
        # If all else fails, truncate the original text
        return text[:250] + "..." if len(text) > 250 else text
    
    def process_movie_data(self, text):
        """Process movie description text to extract all required information"""
        if not text:
            return {}
            
        # Determine if this is a movie query or a movie description
        is_query = self.is_movie_query(text)
        
        result = {
            'movieTitle': self.extract_movie_title(text),
            'genres': self.extract_genres(text),
            'movieDesc': self.extract_movie_description(text),
            'cast': self.extract_cast_crew(text),
            'metadataArr': self.extract_metadata(text),
            'criticsScore': self.extract_critics_sentiment(text),  # Now using the sentiment-based method
            'audienceScore': self.extract_audience_sentiment(text),  # Now using the sentiment-based method
            'audienceVerifiedCount': self.extract_verified_count(text),
            'criticsConsensus': self.extract_summary(text, is_critics=True),
            'audienceConsensus': self.extract_summary(text, is_critics=False),
            'isQuery': is_query
        }
        
        # For critic reviews, we'd need separate text input
        result['criticReviews'] = "Not provided"
        
        return result
    
    def format_for_embedding(self, data):
        """Format the extracted data for embedding according to the specified format"""
        if not data:
            return ""
            
        text_parts = [
            f"{data['movieTitle']}",
            f"{', '.join(data['genres'])}" if data.get('genres') else None,
            f"{data['movieDesc']}" if data.get('movieDesc') else None,
            f"{data['cast']}" if data.get('cast') and data['cast'] != "Not specified" else None,
            f"{', '.join(data['metadataArr'])}" if data.get('metadataArr') else None,
            f"Critics Score: {data['criticsScore']}" if data.get('criticsScore') and data['criticsScore'] != "N/A" else None,
            f"Audience Score: {data['audienceScore']}" if data.get('audienceScore') and data['audienceScore'] != "N/A" else None,
            f"Critics summary regard: {data['criticsConsensus']}" if data.get('criticsConsensus') and data['criticsConsensus'] != "No summary available" else None,
            f"Audience summary regard: {data['audienceConsensus']}" if data.get('audienceConsensus') and data['audienceConsensus'] != "No summary available" else None,
        ]
        
        # Filter out None values and join with semicolons
        formatted_text = "; ".join([part for part in text_parts if part])
        return formatted_text


# Example usage
if __name__ == "__main__":
    processor = MovieTextProcessor()
    
    # Test with a movie description
    movie_description = """
    "The Shawshank Redemption" (1994): This R-rated drama stars Tim Robbins and Morgan Freeman. 
    A powerful tale of hope and redemption spanning over 20 years in Shawshank prison. 
    The film runs 2hr 22min. Critics gave it an exceptional 91% on Rotten Tomatoes, with 
    5,432 verified audience reviews averaging 97%. Many viewers say it's a must-see film
    that changed their lives. Director Frank Darabont masterfully adapts Stephen King's novella.
    """
    
    # Test with a movie query
    movie_query = """
    Recommend me some thriller, sci-fi movies that are highly rated by critics and loved by audiences.
    I prefer films with mind-bending plots and good special effects.
    """
    
    # Process both examples
    description_data = processor.process_movie_data(movie_description)
    query_data = processor.process_movie_data(movie_query)
    
    # Format for embedding
    description_embedding = processor.format_for_embedding(description_data)
    query_embedding = processor.format_for_embedding(query_data)
    
    print("Movie Description Data:")
    for key, value in description_data.items():
        print(f"{key}: {value}")
    
    print("\nFormatted for Embedding (Description):")
    print(description_embedding)
    
    print("\n\nMovie Query Data:")
    for key, value in query_data.items():
        print(f"{key}: {value}")
    
    print("\nFormatted for Embedding (Query):")
    print(query_embedding)