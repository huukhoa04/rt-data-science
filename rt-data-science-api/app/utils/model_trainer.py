import pandas as pd
import numpy as np
import re
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Attempt to import NLTK
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    nltk_available = True
except ImportError:
    nltk_available = False

# Function to ensure NLTK resources are available
def ensure_nltk_resources():
    if nltk_available:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
            
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            print("Downloading NLTK tagger...")
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK wordnet...")
            nltk.download('wordnet', quiet=True)
            
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            print("Downloading NLTK vader lexicon...")
            nltk.download('vader_lexicon', quiet=True)


def safe_convert_to_numeric(value):
    """Safely convert a value to numeric, returning np.nan for invalid conversions"""
    try:
        if pd.isna(value):
            return np.nan
        if isinstance(value, str):
            # Remove commas, dollar signs, etc.
            value = re.sub(r'[^\d.]', '', value)
            if value == '':
                return np.nan
        return float(value)
    except:
        return np.nan


def extract_number(text):
    if pd.isna(text):
        return np.nan
    text = str(text).replace(',', '').replace('+', '')
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    return np.nan


def weighted_mean(series, weights):
    """Compute weighted mean, ignoring NaNs."""
    if series.isnull().all() or weights.isnull().all():
        return np.nan
    return np.average(series.fillna(0), weights=weights.fillna(0))


def compute_weighted_scores(processed_data):
    """Compute weighted scores for each category (director, character, etc.)"""
    print("\n--- Computing Weighted Reference Scores ---")
    
    # Prepare data
    data = processed_data.copy()
    
    # Ensure audience and critic scores are numeric
    for col in ['AudienceScore', 'CriticsScore']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Ensure review counts are numeric
    data['AudienceVerifiedCount'] = data['AudienceVerifiedCount'].apply(extract_number)
    data['CriticReviews'] = data['CriticReviews'].apply(extract_number)
        
    # Filter to include only rows with valid scores and reasonable review counts
    valid_data = data[
        (data['AudienceScore'].notna()) & 
        (data['CriticsScore'].notna()) & 
        (data['AudienceVerifiedCount'] >= 50) & 
        (data['CriticReviews'] >= 10)
    ]
    
    print(f"Computing weighted scores using {len(valid_data)} valid entries")
    
    # Initialize weighted scores dictionary
    weighted_scores = {}
    
    # Process Directors
    if 'Director' in valid_data.columns:
        director_stats = {}
        for director, group in valid_data.groupby('Director'):
            if len(group) >= 2:  # Only include directors with at least 2 films
                director_stats[director] = {
                    'aud': weighted_mean(group['AudienceScore'], group['AudienceVerifiedCount']),
                    'crit': weighted_mean(group['CriticsScore'], group['CriticReviews']),
                    'aud_count': group['AudienceVerifiedCount'].sum(),
                    'crit_count': group['CriticReviews'].sum(),
                    'movie_count': len(group),
                    'movies': []
                }
                # Add movie information
                for _, row in group.iterrows():
                    movie_info = {
                        'title': row['Title'] if 'Title' in row else 'Unknown',
                        'audience_score': row['AudienceScore'],
                        'critics_score': row['CriticsScore'],
                        'year': row['ReleaseDate'].split()[-1] if 'ReleaseDate' in row and isinstance(row['ReleaseDate'], str) and row['ReleaseDate'].strip() and len(row['ReleaseDate'].split()) > 0 else None
                    }
                    director_stats[director]['movies'].append(movie_info)
        weighted_scores['Director'] = director_stats
    
    # Process Characters
    for char_col in [col for col in valid_data.columns if col.startswith('Character')]:
        if char_col not in valid_data.columns:
            continue
        
        if 'Character' not in weighted_scores:
            weighted_scores['Character'] = {}
        
        # Group by character
        for character, group in valid_data.groupby(char_col):
            if character == '' or pd.isna(character):
                continue
                
            if character not in weighted_scores['Character']:
                weighted_scores['Character'][character] = {
                    'aud': weighted_mean(group['AudienceScore'], group['AudienceVerifiedCount']),
                    'crit': weighted_mean(group['CriticsScore'], group['CriticReviews']),
                    'aud_count': group['AudienceVerifiedCount'].sum(),
                    'crit_count': group['CriticReviews'].sum(),
                    'movie_count': len(group),
                    'movies': []
                }
                # Add movie information for each character
                for _, row in group.iterrows():
                    movie_info = {
                        'title': row['Title'] if 'Title' in row else 'Unknown',
                        'audience_score': row['AudienceScore'],
                        'critics_score': row['CriticsScore'],
                        'year': row['ReleaseDate'].split()[-1] if 'ReleaseDate' in row and isinstance(row['ReleaseDate'], str) and row['ReleaseDate'].strip() and len(row['ReleaseDate'].split()) > 0 else None
                    }
                    weighted_scores['Character'][character]['movies'].append(movie_info)
    
    # Process Genres
    for genre_col in [col for col in valid_data.columns if col.startswith('Genre') and col != 'Genres']:
        if genre_col not in valid_data.columns:
            continue
        
        if 'Genre' not in weighted_scores:
            weighted_scores['Genre'] = {}
        
        # Group by genre
        for genre, group in valid_data.groupby(genre_col):
            if genre == '' or pd.isna(genre):
                continue
                
            if genre not in weighted_scores['Genre']:
                weighted_scores['Genre'][genre] = {
                    'aud': weighted_mean(group['AudienceScore'], group['AudienceVerifiedCount']),
                    'crit': weighted_mean(group['CriticsScore'], group['CriticReviews']),
                    'aud_count': group['AudienceVerifiedCount'].sum(),
                    'crit_count': group['CriticReviews'].sum(),
                    'movie_count': len(group)
                }
    
    # Process Ratings
    if 'Rating' in valid_data.columns:
        rating_stats = {}
        for rating, group in valid_data.groupby('Rating'):
            if rating != '' and not pd.isna(rating):
                rating_stats[rating] = {
                    'aud': weighted_mean(group['AudienceScore'], group['AudienceVerifiedCount']),
                    'crit': weighted_mean(group['CriticsScore'], group['CriticReviews']),
                    'aud_count': group['AudienceVerifiedCount'].sum(),
                    'crit_count': group['CriticReviews'].sum(),
                    'movie_count': len(group)
                }
        weighted_scores['Rating'] = rating_stats
    
    # Process Release Dates by decade/year group
    if 'ReleaseDate' in valid_data.columns:
        # Try to extract years from release dates
        years = []
        for date in valid_data['ReleaseDate']:
            if isinstance(date, str):
                # Look for 4-digit year pattern
                year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date)
                if year_match:
                    years.append(int(year_match.group(1)))
                else:
                    years.append(None)
            else:
                years.append(None)
        
        valid_data['ReleaseYear'] = years
        valid_year_data = valid_data[valid_data['ReleaseYear'].notna()]
        
        # Create decade/5-year groups
        release_year_groups = {}
        for year_start in range(1920, 2030, 10):  # Group by decade
            year_end = year_start + 9
            year_group = f"{year_start}-{year_end}"
            
            group = valid_year_data[(valid_year_data['ReleaseYear'] >= year_start) & 
                                  (valid_year_data['ReleaseYear'] <= year_end)]
            
            if len(group) > 0:
                release_year_groups[year_group] = {
                    'aud': weighted_mean(group['AudienceScore'], group['AudienceVerifiedCount']),
                    'crit': weighted_mean(group['CriticsScore'], group['CriticReviews']),
                    'aud_count': len(group),
                    'crit_count': len(group),
                    'movie_count': len(group)
                }
        
        weighted_scores['ReleaseYearGroup'] = release_year_groups
    
    # Process Duration
    if 'Duration' in valid_data.columns:
        # Helper function to parse duration into minutes
        def parse_minutes(dur):
            if pd.isna(dur) or dur == '':
                return None
            
            # Check for hour-minute format (e.g., "1h 30m" or "1h30m")
            hour_min_pattern = re.search(r'(\d+)h\s*(\d*)', str(dur))
            if hour_min_pattern:
                hours = int(hour_min_pattern.group(1))
                minutes = int(hour_min_pattern.group(2)) if hour_min_pattern.group(2) else 0
                return hours * 60 + minutes
            
            # Check for minutes-only format (e.g., "90m" or "90 min" or "90 minutes")
            min_pattern = re.search(r'(\d+)\s*(?:m|min|minutes)', str(dur))
            if min_pattern:
                return int(min_pattern.group(1))
            
            # If it's just a number, assume minutes
            if str(dur).isdigit():
                return int(dur)
            
            return None
        
        # Convert duration to minutes
        valid_data['DurationMinutes'] = valid_data['Duration'].apply(parse_minutes)
        valid_duration_data = valid_data[valid_data['DurationMinutes'].notna()]
        
        # Group by duration ranges
        duration_groups = {}
        for start in range(0, 241, 30):  # 0-30, 31-60, 61-90, etc.
            end = start + 30
            group_name = f"{start+1}-{end}"
            
            group = valid_duration_data[(valid_duration_data['DurationMinutes'] > start) & 
                                      (valid_duration_data['DurationMinutes'] <= end)]
            
            if len(group) > 0:
                duration_groups[group_name] = {
                    'aud': weighted_mean(group['AudienceScore'], group['AudienceVerifiedCount']),
                    'crit': weighted_mean(group['CriticsScore'], group['CriticReviews']),
                    'aud_count': len(group),
                    'crit_count': len(group),
                    'movie_count': len(group)
                }
        
        weighted_scores['DurationGroup'] = duration_groups
    
    # Calculate and save summary statistics
    summary = {
        'directors_count': len(weighted_scores.get('Director', {})),
        'characters_count': len(weighted_scores.get('Character', {})),
        'genres_count': len(weighted_scores.get('Genre', {})),
        'ratings_count': len(weighted_scores.get('Rating', {}))
    }
    
    print("\nWeighted scores summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    
    return weighted_scores


def train_models(data_file, model_output_file='models', test_size=0.2, random_state=42):
    """Train Random Forest models to predict audience and critics scores."""
    print(f"Loading data...")
    try:
        # Load from CSV if data_file is a file name
        if isinstance(data_file, str) and (data_file.endswith('.csv') or data_file.endswith('.txt')):
            df = pd.read_csv(data_file)
        # If data_file is a dataframe, use it directly
        elif isinstance(data_file, pd.DataFrame):
            df = data_file
        else:
            raise ValueError("data_file must be a file path ending in .csv/.txt or a pandas DataFrame")
        
        # Clean up column names
        df.columns = [col.strip() for col in df.columns]
        
        # Check if the dataframe contains the necessary columns
        required_columns = ['AudienceScore', 'CriticsScore']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Process data - make a copy to avoid modifying the original
        processed_data = df.copy()
        
        # Ensure numeric columns are properly formatted
        for col in ['AudienceScore', 'CriticsScore', 'AudienceVerifiedCount', 'CriticReviews']:
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].apply(safe_convert_to_numeric)
        
        # Add ReleaseYear column if it doesn't exist but Year does
        if 'ReleaseYear' not in processed_data.columns and 'Year' in processed_data.columns:
            processed_data['ReleaseYear'] = processed_data['Year'].apply(safe_convert_to_numeric)
        
        # Compute weighted scores for different categories
        print("Computing weighted scores...")
        weighted_scores = compute_weighted_scores(processed_data)
        
        # Add additional statistics for explanation
        genre_metrics = {}
        rating_metrics = {}
        anomaly_stats = {}
        
        # Add genre metrics for statistical confidence
        if 'Genres' in processed_data.columns:
            print("Calculating genre metrics...")
            genre_anomalies = {}
            
            # Pre-process all target columns
            processed_data['AudienceScore'] = processed_data['AudienceScore'].apply(lambda x: min(100, max(0, x)) / 100.0 if pd.notna(x) else np.nan)
            processed_data['CriticsScore'] = processed_data['CriticsScore'].apply(lambda x: min(100, max(0, x)) / 100.0 if pd.notna(x) else np.nan)
            
            # Process genres and calculate statistics
            for i, row in processed_data.iterrows():
                if pd.notna(row['Genres']):
                    genres = [g.strip() for g in str(row['Genres']).split(',')]
                    
                    for genre in genres:
                        if genre not in genre_metrics:
                            genre_metrics[genre] = {
                                'aud_scores': [],
                                'crit_scores': [],
                                'count': 0,
                                'diff': []
                            }
                        
                        if pd.notna(row['AudienceScore']) and pd.notna(row['CriticsScore']):
                            genre_metrics[genre]['aud_scores'].append(row['AudienceScore'])
                            genre_metrics[genre]['crit_scores'].append(row['CriticsScore'])
                            genre_metrics[genre]['count'] += 1
                            genre_metrics[genre]['diff'].append(row['AudienceScore'] - row['CriticsScore'])
            
            # Calculate statistics for each genre
            for genre, data in genre_metrics.items():
                if len(data['aud_scores']) > 0:
                    aud_mean = np.mean(data['aud_scores'])
                    crit_mean = np.mean(data['crit_scores'])
                    data['aud_mean'] = aud_mean
                    data['crit_mean'] = crit_mean
                    data['aud_std'] = np.std(data['aud_scores'])
                    data['crit_std'] = np.std(data['crit_scores'])
                    
                    # Calculate 95% confidence intervals
                    aud_conf_int = (
                        aud_mean - 1.96 * data['aud_std'] / np.sqrt(len(data['aud_scores'])),
                        aud_mean + 1.96 * data['aud_std'] / np.sqrt(len(data['aud_scores']))
                    )
                    crit_conf_int = (
                        crit_mean - 1.96 * data['crit_std'] / np.sqrt(len(data['crit_scores'])),
                        crit_mean + 1.96 * data['crit_std'] / np.sqrt(len(data['crit_scores']))
                    )
                    data['aud_conf_int'] = aud_conf_int
                    data['crit_conf_int'] = crit_conf_int
                    
                    # Add anomaly statistics
                    diff_mean = np.mean(data['diff'])
                    diff_std = np.std(data['diff'])
                    audience_favored = sum(1 for d in data['diff'] if d > 0.15) / len(data['diff']) * 100
                    critics_favored = sum(1 for d in data['diff'] if d < -0.15) / len(data['diff']) * 100
                    
                    if 'genre_anomalies' not in anomaly_stats:
                        anomaly_stats['genre_anomalies'] = {}
                    if 'Genres' not in anomaly_stats['genre_anomalies']:
                        anomaly_stats['genre_anomalies']['Genres'] = {}
                    
                    anomaly_stats['genre_anomalies']['Genres'][genre] = {
                        'diff_mean': diff_mean,
                        'diff_std': diff_std,
                        'audience_favored_pct': audience_favored,
                        'critics_favored_pct': critics_favored
                    }
        
        # Add rating metrics for confidence intervals
        if 'Rating' in processed_data.columns:
            print("Calculating rating metrics...")
            rating_counts = processed_data['Rating'].value_counts()
            
            for rating, count in rating_counts.items():
                rating_data = processed_data[processed_data['Rating'] == rating]
                
                if rating not in rating_metrics:
                    rating_metrics[rating] = {}
                
                aud_scores = rating_data['AudienceScore'].dropna().tolist()
                crit_scores = rating_data['CriticsScore'].dropna().tolist()
                
                if len(aud_scores) > 0 and len(crit_scores) > 0:
                    rating_metrics[rating]['aud_mean'] = np.mean(aud_scores)
                    rating_metrics[rating]['crit_mean'] = np.mean(crit_scores)
                    rating_metrics[rating]['aud_std'] = np.std(aud_scores)
                    rating_metrics[rating]['crit_std'] = np.std(crit_scores)
                    rating_metrics[rating]['count'] = len(aud_scores)
        
        # Store weighted scores and metrics in model_info
        weighted_scores['genre_metrics'] = genre_metrics
        weighted_scores['rating_metrics'] = rating_metrics
        weighted_scores['anomaly_stats'] = anomaly_stats
        
        # Create a mask for valid data points (no NaN in target variables)
        mask = ~(processed_data['AudienceScore'].isna() | processed_data['CriticsScore'].isna())
        print(f"Filtered {sum(mask)} valid data points from {len(processed_data)} total records")
        
        # Prepare features for model training
        print("Preparing features for model training...")
        
        # Handle categorical features
        categorical_features = []
        
        # Check for rating column
        if 'Rating' in processed_data.columns:
            categorical_features.append('Rating')
        
        # Check for genres
        genre_columns = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
                         'Documentary', 'Drama', 'Family', 'Fantasy', 'Horror',
                         'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'Western']
        
        for genre in genre_columns:
            if genre in processed_data.columns:
                categorical_features.append(genre)
        
        # Text features
        text_features = []
        if 'Description' in processed_data.columns:
            text_features.append('Description')
        
        # Numeric features - make sure they exist
        numeric_features = []
        for feature in ['ReleaseYear', 'AudienceVerifiedCount', 'CriticReviews']:
            if feature in processed_data.columns:
                numeric_features.append(feature)
        
        # Duration in minutes
        if 'Duration' in processed_data.columns:
            def parse_minutes(dur):
                if pd.isna(dur):
                    return np.nan
                
                try:
                    hours, minutes = 0, 0
                    hour_match = re.search(r'(\d+)\s*h', str(dur))
                    if hour_match:
                        hours = int(hour_match.group(1))
                    
                    min_match = re.search(r'(\d+)\s*m', str(dur))
                    if min_match:
                        minutes = int(min_match.group(1))
                    
                    return hours * 60 + minutes
                except:
                    return np.nan
            
            processed_data['Duration_Minutes'] = processed_data['Duration'].apply(parse_minutes)
            if 'Duration_Minutes' in processed_data.columns:
                numeric_features.append('Duration_Minutes')
        
        # Debug: Print feature information
        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")
        print(f"Text features: {text_features}")
        
        # Prepare feature list
        feature_columns = []
        feature_columns.extend(numeric_features)
        
        # Generate dummy variables for categorical features
        if categorical_features:
            cat_df = pd.get_dummies(processed_data[categorical_features])
            feature_columns.extend(cat_df.columns.tolist())
            processed_data = pd.concat([processed_data, cat_df], axis=1)
        
        if not feature_columns:
            print("Warning: No valid features found. Using fallback features.")
            processed_data['feature_1'] = 1.0
            feature_columns.append('feature_1')
        
        # Apply the mask to features
        X = processed_data[mask][feature_columns].copy()
        
        # Fill NaN values with appropriate defaults
        fillna_values = {}
        for col in X.columns:
            if col in numeric_features:
                fillna_values[col] = X[col].median()
            else:
                fillna_values[col] = 0
        
        X.fillna(fillna_values, inplace=True)
        
        # Prepare the target variables
        y_audience = processed_data[mask]['AudienceScore'].values
        y_critics = processed_data[mask]['CriticsScore'].values
        
        # Add text features using TF-IDF vectorizer if available
        text_vectorizer = None
        if text_features:
            print("Processing text features...")
            text_vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                min_df=5
            )
            
            # Fit the vectorizer on the text data
            text_data = processed_data[mask][text_features[0]].fillna("").values
            text_matrix = text_vectorizer.fit_transform(text_data)
            
            # Add text features to the feature matrix
            for i in range(text_matrix.shape[1]):
                feature_name = f'text_{i}'
                X[feature_name] = text_matrix[:, i].toarray()
                feature_columns.append(feature_name)
        
        # Split the data into training and testing sets
        X_train, X_test, y_audience_train, y_audience_test, y_critics_train, y_critics_test = train_test_split(
            X, y_audience, y_critics, test_size=test_size, random_state=random_state
        )
        
        # Build preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )
        
        # Debug - Check X data stats
        print(f"Feature shape: {X.shape}")
        
        # Train Random Forest model for audience scores
        print("Training audience score model...")
        audience_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=random_state
        )
        
        # Train Random Forest model for critics scores
        print("Training critics score model...")
        critics_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=random_state
        )
        
        # Fit the models
        audience_model.fit(X_train, y_audience_train)
        critics_model.fit(X_train, y_critics_train)
        
        # Evaluate the models
        audience_pred = audience_model.predict(X_test)
        critics_pred = critics_model.predict(X_test)
        
        # Calculate metrics
        audience_rmse = np.sqrt(mean_squared_error(y_audience_test, audience_pred))
        critics_rmse = np.sqrt(mean_squared_error(y_critics_test, critics_pred))
        audience_r2 = r2_score(y_audience_test, audience_pred)
        critics_r2 = r2_score(y_critics_test, critics_pred)
        
        print(f"Audience Score Model - RMSE: {audience_rmse:.2f}, R²: {audience_r2:.4f}")
        print(f"Critics Score Model - RMSE: {critics_rmse:.2f}, R²: {critics_r2:.4f}")
        
        # Extract and organize feature importances
        feature_importance = {
            'audience': {},
            'critics': {}
        }
        
        # Try to get feature importances
        try:
            # New approach to get feature names safely
            if hasattr(preprocessor, 'get_feature_names_out'):
                try:
                    preprocessed_feature_names = preprocessor.get_feature_names_out()
                except Exception as e:
                    print(f"Warning: Could not extract feature names: {e}")
                    # Fallback to generic feature names
                    preprocessed_feature_names = [f'feature_{i}' for i in range(len(feature_columns))]
            else:
                # Fallback for older sklearn versions
                preprocessed_feature_names = feature_columns
            
            for i, importance in enumerate(audience_model.feature_importances_):
                if i < len(preprocessed_feature_names):
                    feature_name = preprocessed_feature_names[i]
                    feature_importance['audience'][feature_name] = importance
                else:
                    feature_importance['audience'][f'feature_{i}'] = importance
            
            for i, importance in enumerate(critics_model.feature_importances_):
                if i < len(preprocessed_feature_names):
                    feature_name = preprocessed_feature_names[i]
                    feature_importance['critics'][feature_name] = importance
                else:
                    feature_importance['critics'][f'feature_{i}'] = importance
        except Exception as e:
            print(f"Warning: Could not extract feature importance details: {e}")
            # Fallback to using indices
            for i, importance in enumerate(audience_model.feature_importances_):
                feature_importance['audience'][f'feature_{i}'] = importance
            for i, importance in enumerate(critics_model.feature_importances_):
                feature_importance['critics'][f'feature_{i}'] = importance
        
        # Print top features for audience score
        print("\nTop 5 features for audience score:")
        top_audience = sorted(feature_importance['audience'].items(), key=lambda x: x[1], reverse=True)[:5]
        for feature, importance in top_audience:
            print(f"  {feature}: {importance:.4f}")
        
        # Save model info
        model_info = {
            'feature_columns': feature_columns,
            'text_vectorizer': text_vectorizer,
            'fillna_values': fillna_values,
            'feature_importance': feature_importance,
            'model_metrics': {
                'audience_rmse': audience_rmse,
                'critics_rmse': critics_rmse,
                'audience_r2': audience_r2,
                'critics_r2': critics_r2
            },
            'weighted_scores': weighted_scores
        }
        
        # Create directories if they don't exist
        os.makedirs('../data/models', exist_ok=True)
        
        # Save models and model info
        print(f"Saving models to ../rt-data-science-api/app/models/{model_output_file}_*.pkl") 
        with open(f'../rt-data-science-api/app/models/{model_output_file}_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        with open('../rt-data-science-api/app/models/audience_score_model.pkl', 'wb') as f:
            pickle.dump(audience_model, f)
        with open('../rt-data-science-api/app/models/critics_score_model.pkl', 'wb') as f:
            pickle.dump(critics_model, f)
        with open('../rt-data-science-api/app/models/model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        with open('../rt-data-science-api/app/models/weighted_scores.pkl', 'wb') as f:
            pickle.dump(weighted_scores, f)
        print("Models and model info saved successfully!")
        return model_info
    
    except Exception as e:
        print(f"Error training models: {e}")
        import traceback
        traceback.print_exc()
        return None 

def analyze_genre_trends(processed_data):
    """Analyze genre trends over time."""
    print("\n--- Phân tích xu hướng thể loại phim qua các năm ---")
    
    # Extract year from ReleaseDate
    def extract_year(date_str):
        if pd.isna(date_str) or not isinstance(date_str, str):
            return None
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            return int(year_match.group(0))
        return None
    
    data = processed_data.copy()
    
    # Add ReleaseYear column if it doesn't exist
    if 'ReleaseYear' not in data.columns:
        data['ReleaseYear'] = data['ReleaseDate'].apply(extract_year)
    
    # Filter rows with valid release year
    data = data[data['ReleaseYear'].notna()]
    
    # Check if we have any valid data
    if len(data) == 0:
        print("Không đủ dữ liệu năm phát hành để phân tích xu hướng.")
        return None
    
    # Get top 10 genres
    all_genres = []
    for i in range(1, 6):
        col = f'Genre{i}'
        if col in data.columns:
            genres = data[col].dropna().value_counts().index.tolist()
            all_genres.extend(genres)
    
    # Count occurrences of each genre
    genre_counts = Counter(all_genres)
    top_genres = [genre for genre, count in genre_counts.most_common(10)]
    
    # Group data by decade
    data['Decade'] = (data['ReleaseYear'] // 10) * 10
    decades = sorted(data['Decade'].unique())
    
    # If no decades found, return
    if not decades:
        print("Không đủ dữ liệu thập kỷ để phân tích xu hướng.")
        return None
    
    # Calculate genre percentages by decade
    genre_trends = {}
    for genre in top_genres:
        genre_trends[genre] = []
        for decade in decades:
            decade_data = data[data['Decade'] == decade]
            genre_count = 0
            for i in range(1, 6):
                col = f'Genre{i}'
                if col in decade_data.columns:
                    genre_count += (decade_data[col] == genre).sum()
            
            # Calculate percentage
            percentage = genre_count / len(decade_data) * 100 if len(decade_data) > 0 else 0
            genre_trends[genre].append(percentage)
    
    # Print trends
    print(f"Analyzing genre trends from {min(decades)} to {max(decades) + 9}")
    print("\nXu hướng thể loại phim:")
    
    for genre in top_genres:
        if len(genre_trends[genre]) >= 2:
            start_pct = genre_trends[genre][0]
            end_pct = genre_trends[genre][-1]
            pct_change = end_pct - start_pct
            
            # Determine trend
            if pct_change > 10:
                trend = "Tăng mạnh"
            elif pct_change > 5:
                trend = "Tăng"
            elif pct_change < -10:
                trend = "Giảm mạnh"
            elif pct_change < -5:
                trend = "Giảm"
            else:
                trend = "Ổn định"
            
            # Recent trend (last two decades)
            if len(genre_trends[genre]) >= 2:
                recent_change = genre_trends[genre][-1] - genre_trends[genre][-2]
                recent_trend = "tăng" if recent_change > 0 else "giảm"
            else:
                recent_trend = "không xác định"
            
            # Find peak decade
            peak_idx = genre_trends[genre].index(max(genre_trends[genre]))
            peak_decade = decades[peak_idx]
            peak_pct = genre_trends[genre][peak_idx]
            
            print(f"{genre}: {trend} ({start_pct:.1f}% → {end_pct:.1f}%), Xu hướng gần đây: {recent_trend}, Phổ biến nhất vào năm {peak_decade} ({peak_pct:.1f}%)")
    
    # Identify significant changes
    print("\nNhững thay đổi đáng chú ý:")
    for genre in top_genres:
        if len(genre_trends[genre]) < 2:
            continue
        
        for i in range(1, len(decades)):
            prev_pct = genre_trends[genre][i-1]
            curr_pct = genre_trends[genre][i]
            change = curr_pct - prev_pct
            
            if abs(change) > 20:
                direction = "Tăng" if change > 0 else "Giảm"
                print(f"{genre}: {direction} đột biến {abs(change):.1f}% từ năm {decades[i-1]} đến {decades[i]}")
    
    return genre_trends

def analyze_rating_discrepancies(processed_data):
    """Analyze discrepancies between audience and critic scores."""
    print("\n--- Review Discrepancy Analysis ---")
    
    data = processed_data.copy()
    
    # Ensure scores are properly converted to float between 0-1
    if 'AudienceScore' in data.columns and 'CriticsScore' in data.columns:
        data['AudienceScore'] = data['AudienceScore'].apply(lambda x: float(x)/100 if isinstance(x, (int, float)) and x > 1 else x)
        data['CriticsScore'] = data['CriticsScore'].apply(lambda x: float(x)/100 if isinstance(x, (int, float)) and x > 1 else x)
        
        # Filter for valid scores
        data = data[(data['AudienceScore'].notna()) & (data['CriticsScore'].notna())]
        
        # Calculate score difference
        data['ScoreDifference'] = data['AudienceScore'] - data['CriticsScore']
        data['ScoreDifferenceAbs'] = abs(data['ScoreDifference'])
        
        # Calculate basic statistics
        avg_diff = data['ScoreDifference'].mean()
        median_diff = data['ScoreDifference'].median()
        std_diff = data['ScoreDifference'].std()
        
        print(f"Average audience-critics score difference: {avg_diff:.4f}")
        print(f"Median difference: {median_diff:.4f}")
        print(f"Standard deviation: {std_diff:.4f}")
        
        # Analyze by genre
        print("\nAverage score difference by genre:")
        for col in [c for c in data.columns if c.startswith('Genre')]:
            genre_diffs = {}
            for genre in data[col].dropna().unique():
                if genre and not pd.isna(genre):
                    genre_data = data[data[col] == genre]
                    if len(genre_data) > 0:
                        genre_diffs[genre] = (genre_data['ScoreDifference'].mean(), len(genre_data))
            
            for genre, (diff, count) in genre_diffs.items():
                print(f"{genre}: {diff:.4f} ({count} movies)")
        
        # Identify anomalous movies (outliers)
        threshold = std_diff * 1.5
        anomalies = data[(data['ScoreDifferenceAbs'] > threshold)]
        
        print(f"\nIdentifying movies with anomalous rating discrepancies:")
        print(f"Found {len(anomalies)} movies with anomalous rating patterns")
        
        print("\nTop anomalous movies:")
        for _, row in anomalies.iterrows():
            print(f"{row['Title']}: Audience {row['AudienceScore']:.2f}, Critics {row['CriticsScore']:.2f}, Diff {row['ScoreDifference']:.2f}")
        
        return anomalies
    else:
        print("Required columns AudienceScore and CriticsScore not found")
        return None

def perform_genre_clustering(processed_data, n_clusters=5):
    """Perform K-means clustering based on genres and other categorical features."""
    print("\n--- Enhanced K-means Clustering Analysis ---")
    
    # Check if we have enough data
    if processed_data is None or len(processed_data) < n_clusters:
        print(f"Không đủ dữ liệu để phân cụm (cần ít nhất {n_clusters} phim)")
        return None
    
    data = processed_data.copy()
    
    # Identify categorical features for clustering
    cluster_features = []
    
    # Genre columns
    genre_cols = [col for col in data.columns if col.startswith('Genre')]
    if genre_cols:
        cluster_features.extend(genre_cols)
    
    # Director and cast
    if 'Director' in data.columns:
        cluster_features.append('Director')
    
    character_cols = [col for col in data.columns if col.startswith('Character')]
    if character_cols:
        cluster_features.extend(character_cols)
    
    # Check if we have enough features
    if len(cluster_features) < 2:
        print("Không đủ đặc trưng phân loại cho việc phân cụm")
        return None
    
    print(f"Using features for clustering: {cluster_features}")
    
    # Prepare data for clustering
    # Convert categorical variables to one-hot encoding
    for col in cluster_features:
        if col in data.columns:
            # Fill NaN values
            data[col] = data[col].fillna('')
    
    # Create a binary feature matrix
    feature_matrix = pd.get_dummies(data[cluster_features])
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(feature_matrix)
    
    print(f"\nMovie Clusters Analysis (using {n_clusters} clusters):")
    
    for cluster_id in range(n_clusters):
        cluster_data = data[data['Cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_data)} movies):")
        
        # Top directors
        if 'Director' in data.columns:
            top_directors = cluster_data['Director'].value_counts().head(3)
            print("Top Directors:")
            print(top_directors)
            print()
        
        # Top genres
        genre_counts = {}
        for col in genre_cols:
            for genre in cluster_data[col].dropna().value_counts().index:
                if genre != '':
                    if genre not in genre_counts:
                        genre_counts[genre] = 0
                    genre_counts[genre] += cluster_data[col].value_counts()[genre]
        
        print("Top Genres:")
        for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"{genre}: {count}")
        print()
        
        # Average scores
        if 'AudienceScore' in data.columns and 'CriticsScore' in data.columns:
            avg_audience = cluster_data['AudienceScore'].mean()
            avg_critics = cluster_data['CriticsScore'].mean()
            print(f"Average Audience Score: {avg_audience:.2f}")
            print(f"Average Critics Score: {avg_critics:.2f}")
            print()
        
        # Example movies
        print("Example Movies:")
        for title in cluster_data['Title'].head(3):
            print(f"- {title}")
    
    return data['Cluster']

def analyze_movie_data(data_file):
    """Perform comprehensive movie data analysis."""
    results = {
        'processed_data': None,
        'weighted_scores': None,
        'genre_trends': None,
        'anomalies': None,
        'cluster_labels': None
    }
    
    try:
        # Load from CSV if data_file is a file name
        if isinstance(data_file, str) and (data_file.endswith('.csv') or data_file.endswith('.txt')):
            df = pd.read_csv(data_file)
        # If data_file is a dataframe, use it directly
        elif isinstance(data_file, pd.DataFrame):
            df = data_file
        else:
            raise ValueError("data_file must be a file path ending in .csv/.txt or a pandas DataFrame")
        
        # Clean up column names
        df.columns = [col.strip() for col in df.columns]
        
        # Ensure NLTK resources are available
        if nltk_available:
            ensure_nltk_resources()
        
        # Process data - make a copy to avoid modifying the original
        processed_data = df.copy()
        results['processed_data'] = processed_data
        
        # Ensure numeric columns are properly formatted
        for col in ['AudienceScore', 'CriticsScore', 'AudienceVerifiedCount', 'CriticReviews']:
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].apply(safe_convert_to_numeric)
        
        # Add ReleaseYear column if it doesn't exist but Year does
        if 'ReleaseYear' not in processed_data.columns and 'Year' in processed_data.columns:
            processed_data['ReleaseYear'] = processed_data['Year'].apply(safe_convert_to_numeric)
        
        # Print basic dataset info
        print(f"\nDataset columns:")
        print(processed_data.columns.tolist())
        print("\nSample data:")
        print(processed_data.head(3))
        
        # Create output directories
        os.makedirs('../data/models', exist_ok=True)
        os.makedirs('../data/processed', exist_ok=True)
        
        # Perform clustering analysis
        try:
            cluster_labels = perform_genre_clustering(processed_data)
            if cluster_labels is not None:
                processed_data['Cluster'] = cluster_labels
                results['cluster_labels'] = cluster_labels
        except Exception as e:
            print(f"Lỗi trong quá trình phân cụm: {str(e)}")
        
        # Compute weighted scores for different categories
        try:
            print("Computing weighted scores...")
            weighted_scores = compute_weighted_scores(processed_data)
            results['weighted_scores'] = weighted_scores
            
            # Save weighted scores
            with open('../rt-data-science-api/app/models/weighted_scores.pkl', 'wb') as f:
                pickle.dump(weighted_scores, f)
            print("Saved weighted reference scores to ../rt-data-science-api/app/models/weighted_scores.pkl")
        except Exception as e:
            print(f"Lỗi trong quá trình tính điểm trọng số: {str(e)}")
        
        # Analyze genre trends over time
        try:
            genre_trends = analyze_genre_trends(processed_data)
            results['genre_trends'] = genre_trends
        except Exception as e:
            print(f"Lỗi trong quá trình phân tích xu hướng thể loại: {str(e)}")
        
        # Analyze rating discrepancies
        try:
            anomalies = analyze_rating_discrepancies(processed_data)
            results['anomalies'] = anomalies
        except Exception as e:
            print(f"Lỗi trong quá trình phân tích chênh lệch đánh giá: {str(e)}")
        
        # Save processed data
        try:
            processed_data.to_csv('../data/processed/movies_data_processed.csv', index=False)
            print("\nAnalysis completed! Output files saved.")
        except Exception as e:
            print(f"Lỗi khi lưu dữ liệu processed: {str(e)}")
        
        return results
    
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return results 