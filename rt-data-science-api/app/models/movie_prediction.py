import pickle
import pandas as pd
import numpy as np
import re
import os
import traceback
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
import random
import logging

logger = logging.getLogger(__name__)

class MoviePredictor:
    """Movie score prediction model that uses trained models to predict audience and critic scores."""
    
    def __init__(self, 
                 audience_model_path='../models/audience_score_model.pkl',
                 critics_model_path='../models/critics_score_model.pkl',
                 model_info_path='../models/model_info.pkl'):
        """Initialize the movie predictor with paths to the model files."""
        self.audience_model_path = audience_model_path
        self.critics_model_path = critics_model_path
        self.model_info_path = model_info_path
        self.audience_model = None
        self.critics_model = None
        self.model_info = None
        self.vectorizer = None
        self.load_models()
    
    def load_models(self):
        """Load trained prediction models and model info from files"""
        print(f"Loading models from: {self.audience_model_path} and {self.critics_model_path}")
        
        try:
            with open(self.audience_model_path, 'rb') as f:
                self.audience_model = pickle.load(f)
                print("Audience score model loaded successfully")
                
                # Print model info if available
                if hasattr(self.audience_model, 'named_steps'):
                    print(f"Audience model pipeline steps: {list(self.audience_model.named_steps.keys())}")
                    
                    # Check for preprocessing step that might contain feature names
                    if 'preprocessor' in self.audience_model.named_steps:
                        preprocessor = self.audience_model.named_steps['preprocessor']
                        if hasattr(preprocessor, 'get_feature_names_out'):
                            try:
                                feature_names = preprocessor.get_feature_names_out()
                                print(f"Found {len(feature_names)} features in preprocessor")
                            except Exception as e:
                                print(f"Could not get feature names from preprocessor: {e}")
                
            with open(self.critics_model_path, 'rb') as f:
                self.critics_model = pickle.load(f)
                print("Critics score model loaded successfully")
        except Exception as e:
            print(f"Error loading prediction models: {e}")
            # Just continue, as we can use fallback prediction if models aren't available
            
        # Try to load model info if available
        try:
            if os.path.exists(self.model_info_path):
                with open(self.model_info_path, 'rb') as f:
                    self.model_info = pickle.load(f)
                    print("Model info loaded successfully")
                    
                    # Print summary of model info
                    if isinstance(self.model_info, dict):
                        print(f"Model info keys: {list(self.model_info.keys())}")
            
                        # Check for text vectorizer if available
                        if 'vectorizer' in self.model_info:
                            self.vectorizer = self.model_info['vectorizer']
                            print("Text vectorizer found in model info")
                        
                        # Try to load weighted scores separately if not in model_info
                        if 'weighted_scores' not in self.model_info:
                            weighted_scores_path = os.path.join(os.path.dirname(self.model_info_path), 'weighted_scores.pkl')
                            if os.path.exists(weighted_scores_path):
                                try:
                                    with open(weighted_scores_path, 'rb') as f:
                                        weighted_scores = pickle.load(f)
                                        print("Loaded weighted scores separately")
                                        self.model_info['weighted_scores'] = weighted_scores
                                except Exception as e:
                                    print(f"Could not load weighted scores: {e}")
        except Exception as e:
            print(f"Error loading model info: {e}")
        
        return self.audience_model is not None and self.critics_model is not None
    
    def safe_convert_to_numeric(self, value):
        """Safely convert a string that might contain percentage or multiple values to a numeric value"""
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove percentage signs if present
            value = value.replace('%', '')
            
            # Remove other non-numeric characters (except decimal point)
            value = ''.join(c for c in value if c.isdigit() or c == '.')
            
            try:
                return float(value)
            except ValueError:
                pass
        
        return np.nan
    
    def weighted_mean(self, series, weights):
        """Compute weighted mean, ignoring NaNs."""
        if series.isnull().all() or weights.isnull().all():
            return np.nan
        return np.average(series.fillna(0), weights=weights.fillna(0))
    
    def get_weighted_group_score(self, group_name, group_type, weighted_scores):
        """Lookup weighted average score for a director, character, or franchise from processed data."""
        if not weighted_scores or group_type not in weighted_scores:
            return None
        
        if group_type in weighted_scores and group_name in weighted_scores[group_type]:
            scores = weighted_scores[group_type][group_name]
            # If values are already in 0-100 range, use them as is
            # If values are in 0-1 range, convert to 0-100 range
            aud_score = scores.get('aud', 0)
            crit_score = scores.get('crit', 0)
            
            # Check if conversion is needed (if values are in decimal form)
            if aud_score <= 1.0 and crit_score <= 1.0:
                audience_score = aud_score * 100
                critics_score = crit_score * 100
            else:
                audience_score = aud_score
                critics_score = crit_score
                
            return audience_score, critics_score
        
        return None
    
    def get_rating_score(self, rating, weighted_scores):
        """Lookup weighted average score for a movie rating from processed data."""
        if not weighted_scores or 'Rating' not in weighted_scores:
            return None
        
        if rating in weighted_scores['Rating']:
            scores = weighted_scores['Rating'][rating]
            # If values are already in 0-100 range, use them as is
            # If values are in 0-1 range, convert to 0-100 range
            aud_score = scores.get('aud', 0)
            crit_score = scores.get('crit', 0)
            
            # Check if conversion is needed (if values are in decimal form)
            if aud_score <= 1.0 and crit_score <= 1.0:
                audience_score = aud_score * 100
                critics_score = crit_score * 100
            else:
                audience_score = aud_score
                critics_score = crit_score
                
            return audience_score, critics_score
        
        return None
    
    def explain_prediction(self, movie_info: Dict[str, Any], audience_score: float, critics_score: float) -> str:
        """Generate human-readable explanations for predictions with statistical insights"""

        explanation = []
        explanation.append(f"Predicted audience score: {audience_score:.2f}%")
        explanation.append(f"Predicted critics score: {critics_score:.2f}%")
        
        # Calculate score difference and explain consensus
        score_diff = abs(audience_score - critics_score)
        if score_diff < 10:
            explanation.append("\nStrong consensus between critics and audiences")
        elif score_diff < 20:
            explanation.append("\nModerate agreement between critics and audiences")
        else:
            if audience_score > critics_score:
                explanation.append("\nAudiences likely to enjoy this more than critics")
                # Add anomaly explanation for audience-favored films
                anomaly_reasons = []
                
                # Check if genres are likely to cause audience preference
                selected_genres = []
                for genre in ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
                              'Documentary', 'Drama', 'Family', 'Fantasy', 'Horror',
                              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'Western']:
                    if movie_info.get(genre, 0) == 1:
                        selected_genres.append(genre)
                
                # Add genre-based anomaly explanations
                weighted_scores = self.model_info.get('weighted_scores', {}) if isinstance(self.model_info, dict) else {}
                for genre in selected_genres:
                    if weighted_scores and 'Genre' in weighted_scores and genre in weighted_scores['Genre']:
                        genre_data = weighted_scores['Genre'][genre]
                        if genre_data.get('aud', 0) > genre_data.get('crit', 0):
                            # Limit percentage to reasonable values (0-100%)
                            diff = min(100, (genre_data.get('aud', 0) - genre_data.get('crit', 0)) * 100)
                            if diff > 5:  # Only mention if there's a meaningful difference
                                anomaly_reasons.append(f"{genre} films typically score {diff:.1f}% higher with audiences than critics")
                
                # Add rating-based anomaly explanations
                if 'Rating' in movie_info and movie_info['Rating'] and weighted_scores and 'Rating' in weighted_scores:
                    rating = movie_info['Rating']
                    if rating in weighted_scores['Rating']:
                        rating_data = weighted_scores['Rating'][rating]
                        if rating_data.get('aud', 0) > rating_data.get('crit', 0):
                            # Limit percentage to reasonable values (0-100%)
                            diff = min(100, (rating_data.get('aud', 0) - rating_data.get('crit', 0)) * 100)
                            if diff > 5:  # Only mention if there's a meaningful difference
                                anomaly_reasons.append(f"{rating}-rated films typically score {diff:.1f}% higher with audiences than critics")
                
                if anomaly_reasons:
                    explanation.append("\nPossible reasons for audience preference:")
                    for reason in anomaly_reasons[:3]:  # Limit to 3 most significant reasons
                        explanation.append(f"- {reason}")
            else:
                explanation.append("\nCritics likely to rate this higher than general audiences")
                # Add anomaly explanation for critic-favored films
                anomaly_reasons = []
                
                # Check if genres are likely to cause critic preference
                selected_genres = []
                for genre in ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
                              'Documentary', 'Drama', 'Family', 'Fantasy', 'Horror',
                              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'Western']:
                    if movie_info.get(genre, 0) == 1:
                        selected_genres.append(genre)
                
                # Add genre-based anomaly explanations
                weighted_scores = self.model_info.get('weighted_scores', {}) if isinstance(self.model_info, dict) else {}
                for genre in selected_genres:
                    if weighted_scores and 'Genre' in weighted_scores and genre in weighted_scores['Genre']:
                        genre_data = weighted_scores['Genre'][genre]
                        if genre_data.get('crit', 0) > genre_data.get('aud', 0):
                            # Limit percentage to reasonable values (0-100%)
                            diff = min(100, (genre_data.get('crit', 0) - genre_data.get('aud', 0)) * 100)
                            if diff > 5:  # Only mention if there's a meaningful difference
                                anomaly_reasons.append(f"{genre} films typically score {diff:.1f}% higher with critics than audiences")
                
                # Add rating-based anomaly explanations
                if 'Rating' in movie_info and movie_info['Rating'] and weighted_scores and 'Rating' in weighted_scores:
                    rating = movie_info['Rating']
                    if rating in weighted_scores['Rating']:
                        rating_data = weighted_scores['Rating'][rating]
                        if rating_data.get('crit', 0) > rating_data.get('aud', 0):
                            # Limit percentage to reasonable values (0-100%)
                            diff = min(100, (rating_data.get('crit', 0) - rating_data.get('aud', 0)) * 100)
                            if diff > 5:  # Only mention if there's a meaningful difference
                                anomaly_reasons.append(f"{rating}-rated films typically score {diff:.1f}% higher with critics than audiences")
                
                if anomaly_reasons:
                    explanation.append("\nPossible reasons for critic preference:")
                    for reason in anomaly_reasons[:3]:  # Limit to 3 most significant reasons
                        explanation.append(f"- {reason}")
        
        # Add factors that influenced the prediction
        explanation.append("\n=== Statistical Score Breakdown ===")
        
        # Get weighted scores if available in model_info
        weighted_scores = self.model_info.get('weighted_scores', {}) if isinstance(self.model_info, dict) else {}
        
        # Display model Feature importance information if available
        if isinstance(self.model_info, dict) and 'feature_importance' in self.model_info:
            feature_importance = self.model_info['feature_importance']
            explanation.append("\nTop predictive features for this type of movie:")
            
            # Get audience feature importance
            if 'audience' in feature_importance:
                audience_features = feature_importance['audience']
                top_audience = sorted(audience_features.items(), key=lambda x: x[1], reverse=True)[:5]
                explanation.append("Audience score prediction influenced by:")
                for feature, importance in top_audience:
                    # Make feature name more readable
                    readable_feature = feature.replace('_', ' ').title()
                    if feature.startswith('Genre'):
                        readable_feature = "Genre Information"
                    elif feature.startswith('Rating_'):
                        readable_feature = f"Rating: {feature.replace('Rating_', '')}"
                    elif feature.startswith('text_'):
                        readable_feature = "Text Content"
                    # Limit percentage to reasonable values (0-100%)
                    importance_pct = min(100, importance*100)
                    explanation.append(f"  - {readable_feature}: {importance_pct:.1f}% importance")
            
            # Get critics feature importance
            if 'critics' in feature_importance:
                critics_features = feature_importance['critics']
                top_critics = sorted(critics_features.items(), key=lambda x: x[1], reverse=True)[:5]
                explanation.append("\nCritics score prediction influenced by:")
                for feature, importance in top_critics:
                    # Make feature name more readable
                    readable_feature = feature.replace('_', ' ').title()
                    if feature.startswith('Genre'):
                        readable_feature = "Genre Information"
                    elif feature.startswith('Rating_'):
                        readable_feature = f"Rating: {feature.replace('Rating_', '')}"
                    elif feature.startswith('text_'):
                        readable_feature = "Text Content"
                    # Limit percentage to reasonable values (0-100%)
                    importance_pct = min(100, importance*100)
                    explanation.append(f"  - {readable_feature}: {importance_pct:.1f}% importance")
        
        # Director influence
        if 'Director' in movie_info and movie_info['Director'] and weighted_scores and 'Director' in weighted_scores:
            director = movie_info['Director']
            director_score = self.get_weighted_group_score(director, 'Director', weighted_scores)
            if director_score:
                aud_score, crit_score = director_score
                # Limit percentages to 0-100
                aud_score = min(100, max(0, aud_score))
                crit_score = min(100, max(0, crit_score))
                explanation.append(f"\nDirector Analysis ({director}):")
                explanation.append(f"- Previous films averaged {aud_score:.1f}% audience, {crit_score:.1f}% critics")
                
                # Add director review count if available
                if 'Director' in weighted_scores and director in weighted_scores['Director']:
                    dir_data = weighted_scores['Director'][director]
                    if 'aud_count' in dir_data and 'crit_count' in dir_data:
                        explanation.append(f"- Based on {dir_data['aud_count']:.0f} audience reviews and {dir_data['crit_count']:.0f} critic reviews")
                    
                    # Add genre expertise if available
                    if 'genres' in dir_data:
                        dir_genres = dir_data['genres']
                        top_genres = sorted(dir_genres.items(), key=lambda x: x[1], reverse=True)[:3]
                        if top_genres:
                            explanation.append("- Director genre expertise:")
                            for genre, count in top_genres:
                                explanation.append(f"  * {genre}: {count:.0f} films")
        
        # Character influence (check for Character1, Character2, etc.)
        characters_added = 0
        for i in range(1, 11):  # Check up to 10 character fields
            char_key = f'Character{i}'
            if char_key in movie_info and movie_info[char_key] and weighted_scores and 'Character' in weighted_scores:
                character = movie_info[char_key]
                character_score = self.get_weighted_group_score(character, 'Character', weighted_scores)
                if character_score and characters_added < 3:  # Limit to 3 characters to avoid overwhelming
                    aud_score, crit_score = character_score
                    # Limit percentages to 0-100
                    aud_score = min(100, max(0, aud_score))
                    crit_score = min(100, max(0, crit_score))
                    explanation.append(f"\nCharacter Analysis ({character}):")
                    explanation.append(f"- Films with this character averaged {aud_score:.1f}% audience, {crit_score:.1f}% critics")
                    
                    # Add character review count if available
                    if character in weighted_scores['Character']:
                        char_data = weighted_scores['Character'][character]
                        if 'aud_count' in char_data and 'crit_count' in char_data:
                            explanation.append(f"- Based on {char_data['aud_count']:.0f} audience reviews and {char_data['crit_count']:.0f} critic reviews")
                        
                        # Add movie list for this character if available
                        if 'movies' in char_data and char_data['movies']:
                            explanation.append(f"- Notable films with {character}:")
                            for idx, movie in enumerate(char_data['movies']):
                                if idx >= 3:  # Limit to 3 movies
                                    remaining = len(char_data['movies']) - 3
                                    if remaining > 0:
                                        explanation.append(f"  * ...and {remaining} more")
                                    break
                                title = movie.get('title', 'Unknown')
                                a_score = movie.get('audience_score')
                                c_score = movie.get('critics_score')
                                if a_score is not None and c_score is not None:
                                    # Check if scores are already in 0-100 range or 0-1 range
                                    if a_score > 1 or c_score > 1:
                                        # Already in 0-100 range
                                        explanation.append(f"  * {title} (Audience: {a_score:.1f}%, Critics: {c_score:.1f}%)")
                                    else:
                                        # Convert from 0-1 range to percentage
                                        explanation.append(f"  * {title} (Audience: {a_score*100:.1f}%, Critics: {c_score*100:.1f}%)")
                                else:
                                    explanation.append(f"  * {title}")
                    characters_added += 1
        
        # Rating influence with confidence intervals
        if 'Rating' in movie_info and movie_info['Rating'] and weighted_scores:
            rating = movie_info['Rating']
            
            # Check standard weighted scores
            rating_info = self.get_rating_score(rating, weighted_scores)
            if rating_info:
                aud_score, crit_score = rating_info
                # Limit percentages to 0-100
                aud_score = min(100, max(0, aud_score))
                crit_score = min(100, max(0, crit_score))
                explanation.append(f"\nRating Analysis ({rating}):")
                explanation.append(f"- Films with this rating averaged {aud_score:.1f}% audience, {crit_score:.1f}% critics")
                
                # Add rating review count if available
                if 'Rating' in weighted_scores and rating in weighted_scores['Rating']:
                    rating_data = weighted_scores['Rating'][rating]
                    if 'aud_count' in rating_data and 'crit_count' in rating_data:
                        explanation.append(f"- Based on {rating_data['aud_count']:.0f} audience reviews and {rating_data['crit_count']:.0f} critic reviews")
            
            # Add confidence intervals if available in rating_metrics
            if 'rating_metrics' in weighted_scores and rating in weighted_scores['rating_metrics']:
                rating_metrics = weighted_scores['rating_metrics'][rating]
                if 'aud_std' in rating_metrics and 'crit_std' in rating_metrics:
                    # Limit percentages to 0-100
                    explanation.append(f"- Audience score typically ranges from {max(0, min(100, aud_score - rating_metrics['aud_std'])):.1f}% to {min(100, aud_score + rating_metrics['aud_std']):.1f}%")
                    explanation.append(f"- Critics score typically ranges from {max(0, min(100, crit_score - rating_metrics['crit_std'])):.1f}% to {min(100, crit_score + rating_metrics['crit_std']):.1f}%")
        
        # Add genre information and statistics with confidence intervals
        selected_genres = []
        for genre in ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
                      'Documentary', 'Drama', 'Family', 'Fantasy', 'Horror',
                      'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'Western']:
            if movie_info.get(genre, 0) == 1:
                selected_genres.append(genre)
        
        if selected_genres:
            explanation.append(f"\nGenre Analysis:")
            explanation.append(f"- Selected genres: {', '.join(selected_genres)}")
            
            # Check for anomaly analysis in anomaly_stats
            if weighted_scores and 'anomaly_stats' in weighted_scores:
                anomaly_stats = weighted_scores['anomaly_stats']
                if 'genre_anomalies' in anomaly_stats:
                    genre_anomalies = anomaly_stats['genre_anomalies']
                    
                    # Check each genre column
                    for genre_col in genre_anomalies:
                        for genre in selected_genres:
                            if genre in genre_anomalies[genre_col]:
                                anomaly_data = genre_anomalies[genre_col][genre]
                                diff_mean = anomaly_data.get('diff_mean', 0)
                                
                                # Display whether genre tends to favor audience or critics
                                if diff_mean > 5:
                                    # Limit percentages to 0-100
                                    diff_mean = min(100, diff_mean)
                                    audience_favored_pct = min(100, anomaly_data.get('audience_favored_pct', 0))
                                    explanation.append(f"- {genre} films tend to be audience favorites (average {diff_mean:.1f}% higher audience scores)")
                                    explanation.append(f"  * {audience_favored_pct:.1f}% of {genre} films are strongly audience-favored")
                                elif diff_mean < -5:
                                    # Limit percentages to 0-100
                                    diff_mean = min(100, abs(diff_mean))
                                    critics_favored_pct = min(100, anomaly_data.get('critics_favored_pct', 0))
                                    explanation.append(f"- {genre} films tend to be critic favorites (average {diff_mean:.1f}% higher critic scores)")
                                    explanation.append(f"  * {critics_favored_pct:.1f}% of {genre} films are strongly critic-favored")
            
            # Add detailed genre statistics if available from genre_metrics
            if weighted_scores and 'genre_metrics' in weighted_scores:
                genre_metrics = weighted_scores['genre_metrics']
                explanation.append("\nGenre Historical Statistics:")
                for genre in selected_genres:
                    if genre in genre_metrics:
                        metrics = genre_metrics[genre]
                        explanation.append(f"- {genre} films ({metrics.get('count', 0)} analyzed):")
                        
                        # Add mean scores with confidence intervals
                        # Limit percentages to 0-100 and convert from decimal if needed
                        aud_mean = metrics.get('aud_mean', 0)
                        crit_mean = metrics.get('crit_mean', 0)
                        aud_std = metrics.get('aud_std', 0)
                        crit_std = metrics.get('crit_std', 0)
                        
                        # Check if values are in 0-1 decimal range and convert if needed
                        if aud_mean <= 1.0 and crit_mean <= 1.0:
                            aud_mean = aud_mean * 100
                            crit_mean = crit_mean * 100
                            aud_std = aud_std * 100
                            crit_std = crit_std * 100
                        
                        # Apply limits 
                        aud_mean = min(100, max(0, aud_mean))
                        crit_mean = min(100, max(0, crit_mean))
                        aud_std = min(50, aud_std)  # Limit std to reasonable value
                        crit_std = min(50, crit_std)  # Limit std to reasonable value
                        
                        explanation.append(f"  * Audience: {aud_mean:.1f}% (±{aud_std:.1f}%)")
                        explanation.append(f"  * Critics: {crit_mean:.1f}% (±{crit_std:.1f}%)")
                        
                        # Add confidence intervals as ranges
                        if 'aud_conf_int' in metrics and 'crit_conf_int' in metrics:
                            aud_low, aud_high = metrics['aud_conf_int']
                            crit_low, crit_high = metrics['crit_conf_int']
                            
                            # Convert to percentages if in decimal form
                            if aud_low <= 1.0 and aud_high <= 1.0:
                                aud_low *= 100
                                aud_high *= 100
                                crit_low *= 100
                                crit_high *= 100
                            
                            # Apply limits
                            aud_low = max(0, min(100, aud_low))
                            aud_high = min(100, aud_high)
                            crit_low = max(0, min(100, crit_low))
                            crit_high = min(100, crit_high)
                            
                            explanation.append(f"  * 95% confidence: Audience scores between {aud_low:.1f}% and {aud_high:.1f}%")
                            explanation.append(f"  * 95% confidence: Critic scores between {crit_low:.1f}% and {crit_high:.1f}%")
                    
                    # If not in metrics but in regular weighted scores
                    elif weighted_scores and 'Genre' in weighted_scores and genre in weighted_scores['Genre']:
                        genre_score = self.get_weighted_group_score(genre, 'Genre', weighted_scores)
                        if genre_score:
                            aud_score, crit_score = genre_score
                            # Limit percentages to 0-100
                            aud_score = min(100, max(0, aud_score))
                            crit_score = min(100, max(0, crit_score))
                            explanation.append(f"- {genre}: Films average {aud_score:.1f}% audience, {crit_score:.1f}% critics")
        
        # Add year statistics if available
        if 'Year' in movie_info and weighted_scores and 'ReleaseYearGroup' in weighted_scores:
            year = self.safe_convert_to_numeric(movie_info['Year'])
            # Find decade or closest year group
            for year_group in sorted(weighted_scores['ReleaseYearGroup'].keys()):
                year_start, year_end = year_group.split('-')
                if int(year_start) <= year <= int(year_end):
                    year_data = weighted_scores['ReleaseYearGroup'][year_group]
                    explanation.append(f"\nTime Period Analysis:")
                    # Limit percentages to 0-100
                    aud_score = min(100, year_data['aud']*100)
                    crit_score = min(100, year_data['crit']*100)
                    explanation.append(f"- Films from {year_group} averaged {aud_score:.1f}% audience, {crit_score:.1f}% critics")
                    explanation.append(f"- Based on {year_data.get('aud_count', 0):.0f} movies from this time period")
                    break
        
        # Add overall model confidence and statistics
        try:
            # If we have model quality metrics in model_info, display them
            if isinstance(self.model_info, dict) and 'model_metrics' in self.model_info:
                metrics = self.model_info['model_metrics']
                explanation.append("\nModel Performance Statistics:")
                if 'audience_r2' in metrics:
                    # Convert R² from decimal to percentage if needed
                    audience_r2 = metrics['audience_r2']
                    audience_rmse = metrics.get('audience_rmse', 15)
                    
                    # Convert R² to percentage if it's small (e.g., 0.06 should be 6%)
                    if audience_r2 < 1.0:  # Likely in decimal form
                        audience_r2 = audience_r2 * 100
                    
                    # Check if RMSE needs conversion (if it's very small compared to expected scale)
                    if audience_rmse < 1.0 and max(audience_score, critics_score) > 50:
                        audience_rmse = audience_rmse * 100
                    
                    # Apply limits
                    audience_r2 = min(100, max(0, audience_r2))
                    audience_rmse = min(50, audience_rmse)  # Limit RMSE to reasonable value
                    
                    explanation.append(f"- Audience score prediction accuracy: {audience_r2:.1f}%")
                    explanation.append(f"- Average audience score error: ±{audience_rmse:.1f}%")
                    
                if 'critics_r2' in metrics:
                    # Convert R² from decimal to percentage if needed
                    critics_r2 = metrics['critics_r2']
                    critics_rmse = metrics.get('critics_rmse', 15)
                    
                    # Convert R² to percentage if it's small (e.g., 0.06 should be 6%)
                    if critics_r2 < 1.0:  # Likely in decimal form
                        critics_r2 = critics_r2 * 100
                    
                    # Check if RMSE needs conversion (if it's very small compared to expected scale)
                    if critics_rmse < 1.0 and max(audience_score, critics_score) > 50:
                        critics_rmse = critics_rmse * 100
                    
                    # Apply limits
                    critics_r2 = min(100, max(0, critics_r2))
                    critics_rmse = min(50, critics_rmse)  # Limit RMSE to reasonable value
                    
                    explanation.append(f"- Critics score prediction accuracy: {critics_r2:.1f}%")
                    explanation.append(f"- Average critics score error: ±{critics_rmse:.1f}%")
        except Exception as e:
            print(f"Error displaying model metrics: {e}")
            pass  # Silently continue if metrics aren't available
        
        return '\n'.join(explanation)
    
    def predict(self, movie_info: Dict[str, Any]) -> Dict[str, Any]:
        """Predict audience and critics scores for a movie based on its attributes."""
        if not movie_info:
            return {
                "error": "No movie information provided",
                "audience_score": None,
                "critics_score": None,
                "explanation": "No movie information provided for prediction."
            }
        
        # Load models if not already loaded
        if not self.audience_model or not self.critics_model:
            if not self.load_models():
                return self.fallback_prediction(movie_info)
        
        try:
            print(f"\n--- Predicting scores for movie: {movie_info.get('Title', 'Unknown Movie')} ---")
            
            # Get weighted scores from model_info
            weighted_scores = self.model_info.get('weighted_scores', {}) if isinstance(self.model_info, dict) else {}
            
            # Debug: Print genres selected for prediction
            selected_genres = []
            for genre in ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
                          'Documentary', 'Drama', 'Family', 'Fantasy', 'Horror',
                          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'Western']:
                if movie_info.get(genre, 0) == 1:
                    selected_genres.append(genre)
            
            print(f"Selected genres: {selected_genres}")
            
            # Process features for prediction
            try:
                # Initialize a dictionary to hold all feature values
                feature_dict = {}
                
                # Load feature columns from model_info
                if isinstance(self.model_info, dict) and 'feature_columns' in self.model_info:
                    expected_columns = self.model_info['feature_columns']
                    # Initialize with default values (0 for numeric features, empty for text)
                    feature_dict = {col: 0 for col in expected_columns}
                else:
                    # Fallback to most common features if model_info doesn't have feature columns
                    feature_dict = {
                        'ReleaseYear': 0,
                        'Duration_Minutes': 0,
                        'AudienceVerifiedCount': 100,
                        'CriticReviews': 100
                    }
                    # Add genre columns
                    for genre in ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
                                'Documentary', 'Drama', 'Family', 'Fantasy', 'Horror',
                                'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'Western']:
                        feature_dict[genre] = 0
                
                # Add Year/ReleaseYear if available
                if 'Year' in movie_info:
                    year_value = self.safe_convert_to_numeric(movie_info['Year'])
                    # Try different column names that might exist in the model
                    for year_col in ['ReleaseYear', 'Year']:
                        if year_col in feature_dict:
                            feature_dict[year_col] = year_value
                            print(f"Set {year_col} = {year_value}")
                
                # Get rating
                selected_rating = None
                # Handle rating as one-hot encoded (Rating_PG, Rating_R, etc.)
                if 'Rating' in movie_info and movie_info['Rating']:
                    rating_value = movie_info['Rating']
                    selected_rating = rating_value
                    rating_found = False
                    
                    # Check if direct rating column exists
                    if 'Rating' in feature_dict:
                        feature_dict['Rating'] = rating_value
                        rating_found = True
                        print(f"Set Rating = {rating_value}")
                    else:
                        # Find all rating columns that match this rating
                        rating_cols = [col for col in feature_dict if col.startswith('Rating_')]
                        if rating_cols:
                            rating_found = False
                            for col in rating_cols:
                                # Extract the rating from the column name
                                col_rating = col.replace('Rating_', '')
                                # Set to 1 if this is the selected rating
                                if col_rating.lower() == rating_value.lower() or col_rating.replace('-', '') == rating_value.replace('-', ''):
                                    feature_dict[col] = 1
                                    rating_found = True
                                    print(f"Set {col} = 1")
                                else:
                                    feature_dict[col] = 0
                                
                    if not rating_found:
                        print(f"Warning: Rating '{rating_value}' not found in model features")
                
                # Process genre columns (Action, Adventure, etc.) as binary indicators (0/1)
                genres_found = 0
                
                # Map API genre names to model genre names (handling special cases)
                genre_mapping = {
                    'SciFi': 'Sci-Fi',
                    'Sci-Fi': 'Sci-Fi',
                    'Science Fiction': 'Sci-Fi'
                }
                
                # Set all genre columns to 0 first
                genre_columns = [col for col in feature_dict if col in [
                    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
                    'Documentary', 'Drama', 'Family', 'Fantasy', 'Horror',
                    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'Western'
                ]]
                for col in genre_columns:
                    feature_dict[col] = 0
                
                # Process direct genre values from input
                for genre in [
                    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
                    'Documentary', 'Drama', 'Family', 'Fantasy', 'Horror',
                    'Musical', 'Mystery', 'Romance', 'SciFi', 'Sci-Fi', 'Thriller', 'Western'
                ]:
                    # Check if this genre is set to 1 in the input
                    if genre in movie_info and movie_info[genre] == 1:
                        # Map genre if needed
                        model_genre = genre_mapping.get(genre, genre)
                        
                        # Check if the model has this genre column
                        if model_genre in feature_dict:
                            feature_dict[model_genre] = 1
                            genres_found += 1
                            print(f"Set {model_genre} = 1 (direct genre column)")
                
                # If no genres were found, try to extract from the direct genre fields
                selected_genres = []
                for genre in [
                    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
                    'Documentary', 'Drama', 'Family', 'Fantasy', 'Horror',
                    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'Western'
                ]:
                    if movie_info.get(genre, 0) == 1 or movie_info.get(genre_mapping.get(genre, ''), 0) == 1:
                        selected_genres.append(genre)
                
                print(f"Selected genres: {selected_genres}")
                
                # Check for one-hot encoded genre columns (e.g., cat__Genre_Drama)
                encoded_genre_cols = [col for col in feature_dict if any(f"__{genre}" in col or f"_{genre}" in col for genre in selected_genres)]
                for col in encoded_genre_cols:
                    for genre in selected_genres:
                        if f"__{genre}" in col or f"_{genre}" in col:
                            feature_dict[col] = 1
                            genres_found += 1
                            print(f"Set {col} = 1 (encoded genre)")
                
                if genres_found == 0 and selected_genres:
                    print(f"Warning: None of the selected genres {selected_genres} found in model features")
                
                # Get characters
                selected_characters = []
                for i in range(1, 11):  # Check up to 10 character fields
                    char_key = f'Character{i}'
                    if char_key in movie_info and movie_info[char_key]:
                        selected_characters.append(movie_info[char_key])
                
                # Process text features (Description) if available
                if 'Description' in feature_dict and 'Synopsis' in movie_info:
                    text_features = movie_info.get('Synopsis', '')
                    feature_dict['Description'] = text_features
                    print(f"Set Description from synopsis: {text_features[:50]}...")
                    
                    # If the model used CountVectorizer, apply it to generate expected text features
                    if self.vectorizer is not None and hasattr(self.vectorizer, 'transform'):
                        try:
                            text_matrix = self.vectorizer.transform([text_features])
                            # Add all text features at once
                            for i, col_name in enumerate(self.vectorizer.get_feature_names_out()):
                                feature_name = f'text_{i}'
                                if feature_name in feature_dict:
                                    feature_dict[feature_name] = text_matrix[0, i]
                            print(f"Added text features")
                        except Exception as e:
                            print(f"Warning: Could not transform text features: {e}")
                
                # Add duration information if needed
                if 'Duration_Minutes' in feature_dict and 'Duration' in movie_info:
                    # Parse duration from format like "2h15m"
                    duration_str = movie_info['Duration']
                    duration_minutes = 0
                    
                    # Extract hours
                    hours_match = re.search(r'(\d+)h', duration_str)
                    if hours_match:
                        duration_minutes += int(hours_match.group(1)) * 60
                    
                    # Extract minutes
                    minutes_match = re.search(r'(\d+)m', duration_str)
                    if minutes_match:
                        duration_minutes += int(minutes_match.group(1))
                    
                    feature_dict['Duration_Minutes'] = duration_minutes
                    print(f"Set Duration_Minutes = {duration_minutes}")
                
                # Handle other numeric features
                for col in ['AudienceVerifiedCount', 'CriticReviews']:
                    if col in feature_dict:
                        # Use a reasonable default value for prediction
                        feature_dict[col] = 100
                        print(f"Set {col} = 100 (default value)")
                
                # Fill missing values based on model info
                if isinstance(self.model_info, dict) and 'fillna_values' in self.model_info:
                    for col in feature_dict:
                        if col in self.model_info['fillna_values'] and (col not in feature_dict or pd.isna(feature_dict[col])):
                            feature_dict[col] = self.model_info['fillna_values'][col]
                            print(f"Filled missing value for {col} = {self.model_info['fillna_values'][col]}")
                
                # Print feature summary for debugging
                print("\nDebug - Feature values summary:")
                non_zero_features = {k: v for k, v in feature_dict.items() if v != 0 and not pd.isna(v) and not (isinstance(v, str) and not v)}
                if non_zero_features:
                    print("Non-zero feature values:")
                    for feat, val in non_zero_features.items():
                        if isinstance(val, str):
                            print(f"  {feat}: <text content>")
                        else:
                            print(f"  {feat}: {val}")
                
                # Print expected feature columns for debugging
                expected_cols = self.model_info.get('feature_columns', []) if isinstance(self.model_info, dict) else []
                print("\nDebug - Available feature columns in model:")
                if expected_cols:
                    print(f"Total expected columns: {len(expected_cols)}")
                    # Print a few sample columns
                    print(f"Sample columns: {expected_cols[:5]}...")
                
                # Create the DataFrame with the exact expected columns
                # This is critical to avoid the feature mismatch error with scikit-learn models
                if expected_cols:
                    # Create DataFrame with only the expected columns in the original order
                    # Previously this way created a mismatch in column order
                    # features = pd.DataFrame({col: [feature_dict.get(col, 0)] for col in expected_cols})
                    
                    # New approach - create a strictly ordered DataFrame that guarantees column order
                    data = []
                    for col in expected_cols:
                        data.append(feature_dict.get(col, 0))
                    features = pd.DataFrame([data], columns=expected_cols)
                else:
                    # If we don't have the expected columns list, create from feature_dict
                    features = pd.DataFrame([feature_dict])
                
                # Convert all data to numeric (float) where possible
                for col in features.columns:
                    # Skip text feature columns
                    if col in features and not (col.startswith('text_') or col == 'Description'):
                        features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
                
                # Make predictions with the model
                try:
                    # Force features to have exactly same columns and order as expected
                    if hasattr(self.audience_model, 'feature_names_in_'):
                        expected_feature_names = self.audience_model.feature_names_in_
                        if set(expected_feature_names) != set(features.columns) or list(expected_feature_names) != list(features.columns):
                            print(f"Warning: Feature names mismatch. Reordering columns to match model expectations.")
                            # Create new dataframe with exact expected columns in exact order
                            ordered_features = pd.DataFrame(columns=expected_feature_names)
                            
                            # Copy data from original features to ordered_features
                            for col in expected_feature_names:
                                if col in features.columns:
                                    ordered_features[col] = features[col]
                                else:
                                    ordered_features[col] = 0
                            
                            features = ordered_features
                    
                    # Execute model prediction with properly ordered features
                    audience_prediction = self.audience_model.predict(features)[0]
                    critics_prediction = self.critics_model.predict(features)[0]
                    
                    # Debug: Print raw prediction values before conversion
                    print(f"\nRaw predictions: Audience {audience_prediction}, Critics {critics_prediction}")
                    
                    # Convert from 0-1 scale to 0-100 percentage scale if predictions are too low
                    if audience_prediction < 1.0 and critics_prediction < 1.0:
                        print("Predictions appear to be in 0-1 range, converting to percentage (0-100)")
                        audience_prediction = audience_prediction * 100
                        critics_prediction = critics_prediction * 100
                except Exception as e:
                    print(f"Error during model prediction: {e}")
                    if "feature names" in str(e) or "X has" in str(e):
                        # This is likely a feature mismatch error - try with direct inputs
                        print("Attempting alternative approach with direct model input...")
                        try:
                            # Try with numpy array instead of DataFrame
                            if hasattr(self.audience_model, 'n_features_in_'):
                                n_features = self.audience_model.n_features_in_
                                print(f"Model requires {n_features} features")
                                
                                # Create array with correct shape
                                x_array = np.zeros((1, n_features))
                                
                                # Try to map basic features into the array
                                feature_importance = {}
                                if isinstance(self.model_info, dict) and 'feature_importance' in self.model_info:
                                    feature_importance = self.model_info['feature_importance']
                                
                                # Set important genre features
                                for genre in selected_genres:
                                    # Try to find genre in feature importance
                                    for i, feat in enumerate(expected_cols):
                                        if genre.lower() in feat.lower() and i < n_features:
                                            x_array[0, i] = 1
                                            print(f"Set feature {i} ({feat}) = 1 for genre {genre}")
                                
                                # Add year if available
                                if 'Year' in movie_info:
                                    year_value = self.safe_convert_to_numeric(movie_info['Year'])
                                    # Try to find year column index
                                    for i, feat in enumerate(expected_cols):
                                        if 'year' in feat.lower() and i < n_features:
                                            x_array[0, i] = year_value
                                            print(f"Set feature {i} ({feat}) = {year_value}")
                                
                                # Run prediction with array
                                audience_prediction = self.audience_model.predict(x_array)[0]
                                critics_prediction = self.critics_model.predict(x_array)[0]
                                print(f"Raw array-based predictions: Audience {audience_prediction}, Critics {critics_prediction}")
                                
                                # Apply genre influence - add fixed influence regardless of genre count
                                modifier = 0.125  # Fixed influence for all genres combined
                                audience_prediction += modifier
                                critics_prediction += modifier
                                print(f"Added genre influence: +{modifier} (fixed value)")
                                
                                # Apply director influence (higher than genre)
                                if 'Director' in movie_info and movie_info['Director'] and 'Director' in weighted_scores:
                                    director = movie_info['Director']
                                    if director in weighted_scores['Director']:
                                        dir_data = weighted_scores['Director'][director]
                                        aud_score = dir_data.get('aud', 0)
                                        crit_score = dir_data.get('crit', 0)
                                        
                                        # Convert to percentage if needed
                                        if aud_score <= 1.0:
                                            aud_score *= 100
                                        if crit_score <= 1.0:
                                            crit_score *= 100
                                        
                                        # Higher weight for director with extreme scores
                                        base_modifier = 0.15
                                        if aud_score > 80 or crit_score > 80 or aud_score < 30 or crit_score < 30:
                                            director_modifier = 0.3  # Double influence
                                            print(f"Doubled director influence due to extreme scores: {aud_score:.1f}% | {crit_score:.1f}%")
                                        else:
                                            director_modifier = base_modifier
                                            
                                        audience_prediction += director_modifier
                                        critics_prediction += director_modifier
                                        print(f"Added director influence: +{director_modifier}")
                                
                                # Apply character influence (equal to genre, fixed value)
                                if any(movie_info.get(f'Character{i}') for i in range(1, 6)):
                                    # Check for extreme character scores
                                    character_extreme_scores = False
                                    for i in range(1, 6):
                                        char_key = f'Character{i}'
                                        if char_key in movie_info and movie_info[char_key] and 'Character' in weighted_scores:
                                            character = movie_info[char_key]
                                            if character in weighted_scores['Character']:
                                                char_data = weighted_scores['Character'][character]
                                                aud_score = char_data.get('aud', 0)
                                                crit_score = char_data.get('crit', 0)
                                                
                                                # Convert to percentage if needed
                                                if aud_score <= 1.0:
                                                    aud_score *= 100
                                                if crit_score <= 1.0:
                                                    crit_score *= 100
                                                
                                                if aud_score > 80 or crit_score > 80 or aud_score < 30 or crit_score < 30:
                                                    character_extreme_scores = True
                                                    print(f"Character '{character}' has extreme scores: {aud_score:.1f}% | {crit_score:.1f}%")
                                                    break
                                    
                                    base_modifier = 0.125
                                    if character_extreme_scores:
                                        char_modifier = 0.25  # Double influence
                                        print(f"Doubled character influence due to extreme scores")
                                    else:
                                        char_modifier = base_modifier
                                        
                                    audience_prediction += char_modifier
                                    critics_prediction += char_modifier
                                    print(f"Added character influence: +{char_modifier}")
                                
                                # Apply rating influence
                                if selected_rating in ['PG', 'G']:
                                    audience_prediction += 0.05
                                elif selected_rating in ['R', 'NC-17']:
                                    critics_prediction += 0.05
                                
                                # Convert to percentage
                                if audience_prediction < 1:
                                    audience_prediction = audience_prediction * 100
                                if critics_prediction < 1:
                                    critics_prediction = critics_prediction * 100
                                
                                print("Successfully used direct model prediction approach")
                            else:
                                # Cannot use direct prediction, fall back
                                return self.fallback_prediction(movie_info)
                        except Exception as inner_e:
                            print(f"Alternative approach failed: {inner_e}")
                            return self.fallback_prediction(movie_info)
                    else:
                        return self.fallback_prediction(movie_info)
                
                # Check for fixed/unrealistic values and use statistics-based approach instead
                model_seems_broken = False
                if abs(audience_prediction - 87.6) < 0.1 and abs(critics_prediction - 43.9) < 0.1:
                    print("WARNING: Model is returning fixed values")
                    model_seems_broken = True
                elif audience_prediction < 5 or critics_prediction < 5:  
                    print("WARNING: Model is returning unrealistically low values")
                    model_seems_broken = True
                
                # Calculate genre statistics to compare with model predictions
                genre_audience_avg = 0
                genre_critics_avg = 0
                genres_found_in_stats = 0
                
                # Initialize aggregation variables
                stats_sources = []
                weighted_audience_sum = 0
                weighted_critics_sum = 0
                total_weight = 0
                
                # Check genre_metrics for more detailed stats
                if 'genre_metrics' in weighted_scores:
                    for genre in selected_genres:
                        if genre in weighted_scores['genre_metrics']:
                            metrics = weighted_scores['genre_metrics'][genre]
                            
                            # Get mean scores
                            aud_mean = metrics.get('aud_mean', 0)
                            crit_mean = metrics.get('crit_mean', 0)
                            
                            # Convert if in 0-1 range
                            if aud_mean <= 1.0 and crit_mean <= 1.0:
                                aud_mean *= 100
                                crit_mean *= 100
                            
                            genre_audience_avg += aud_mean
                            genre_critics_avg += crit_mean
                            genres_found_in_stats += 1
                
                # Calculate average for genre if found
                if genres_found_in_stats > 0:
                    genre_audience_avg /= genres_found_in_stats
                    genre_critics_avg /= genres_found_in_stats
                    
                    # Add to weighted sum (genre weight is fixed at 2.5 regardless of count)
                    weight = 2.5
                    stats_sources.append(f"Genre (weight: {weight:.1f})")
                    weighted_audience_sum += genre_audience_avg * weight
                    weighted_critics_sum += genre_critics_avg * weight
                    total_weight += weight
                    
                    print(f"Genre statistics - Audience avg: {genre_audience_avg:.1f}%, Critics avg: {genre_critics_avg:.1f}%")
                
                # Add director statistics
                director_audience_avg = 0
                director_critics_avg = 0
                director_found = False
                
                if 'Director' in movie_info and movie_info['Director'] and 'Director' in weighted_scores:
                    director = movie_info['Director']
                    if director in weighted_scores['Director']:
                        dir_data = weighted_scores['Director'][director]
                        aud_mean = dir_data.get('aud', 0)
                        crit_mean = dir_data.get('crit', 0)
                        
                        # Convert if in 0-1 range
                        if aud_mean <= 1.0 and crit_mean <= 1.0:
                            aud_mean *= 100
                            crit_mean *= 100
                        
                        director_audience_avg = aud_mean
                        director_critics_avg = crit_mean
                        director_found = True
                        
                        # Add to weighted sum (highest weight for director)
                        # Tăng trọng số nếu đạo diễn có điểm rất cao hoặc rất thấp
                        base_weight = 4.0
                        if director_audience_avg > 80 or director_critics_avg > 80 or director_audience_avg < 30 or director_critics_avg < 30:
                            weight = 8.0  # Tăng gấp đôi khi điểm rất cao hoặc rất thấp
                            print(f"Increased director weight due to extreme scores: {director_audience_avg:.1f}% | {director_critics_avg:.1f}%")
                        else:
                            weight = base_weight
                        
                        stats_sources.append(f"Director (weight: {weight:.1f})")
                        weighted_audience_sum += director_audience_avg * weight
                        weighted_critics_sum += director_critics_avg * weight
                        total_weight += weight
                        
                        print(f"Director statistics - Audience avg: {director_audience_avg:.1f}%, Critics avg: {director_critics_avg:.1f}%")
                
                # Add character statistics
                character_audience_avg = 0
                character_critics_avg = 0
                characters_found = 0
                
                for i in range(1, 11):  # Check up to 10 character fields
                    char_key = f'Character{i}'
                    if char_key in movie_info and movie_info[char_key] and 'Character' in weighted_scores:
                        character = movie_info[char_key]
                        if character in weighted_scores['Character']:
                            char_data = weighted_scores['Character'][character]
                            aud_mean = char_data.get('aud', 0)
                            crit_mean = char_data.get('crit', 0)
                            
                            # Convert if in 0-1 range
                            if aud_mean <= 1.0 and crit_mean <= 1.0:
                                aud_mean *= 100
                                crit_mean *= 100
                            
                            character_audience_avg += aud_mean
                            character_critics_avg += crit_mean
                            characters_found += 1
                
                if characters_found > 0:
                    character_audience_avg /= characters_found
                    character_critics_avg /= characters_found
                    
                    # Add to weighted sum (equal to genre weight, fixed at 2.5 regardless of count)
                    # Tăng trọng số nếu nhân vật có điểm rất cao hoặc rất thấp
                    base_weight = 2.5
                    if character_audience_avg > 80 or character_critics_avg > 80 or character_audience_avg < 30 or character_critics_avg < 30:
                        weight = 5.0  # Tăng gấp đôi khi điểm rất cao hoặc rất thấp
                        print(f"Increased character weight due to extreme scores: {character_audience_avg:.1f}% | {character_critics_avg:.1f}%")
                    else:
                        weight = base_weight
                    
                    stats_sources.append(f"Characters (weight: {weight:.1f})")
                    weighted_audience_sum += character_audience_avg * weight
                    weighted_critics_sum += character_critics_avg * weight
                    total_weight += weight
                    
                    print(f"Character statistics - Audience avg: {character_audience_avg:.1f}%, Critics avg: {character_critics_avg:.1f}%")
                
                # Add rating statistics
                rating_audience_avg = 0
                rating_critics_avg = 0
                rating_found = False
                
                if 'Rating' in movie_info and movie_info['Rating'] and 'Rating' in weighted_scores:
                    rating = movie_info['Rating']
                    if rating in weighted_scores['Rating']:
                        rating_data = weighted_scores['Rating'][rating]
                        aud_mean = rating_data.get('aud', 0)
                        crit_mean = rating_data.get('crit', 0)
                        
                        # Convert if in 0-1 range
                        if aud_mean <= 1.0 and crit_mean <= 1.0:
                            aud_mean *= 100
                            crit_mean *= 100
                        
                        rating_audience_avg = aud_mean
                        rating_critics_avg = crit_mean
                        rating_found = True
                        
                        # Add to weighted sum (lower weight for rating)
                        weight = 1.5
                        stats_sources.append(f"Rating (weight: {weight:.1f})")
                        weighted_audience_sum += rating_audience_avg * weight
                        weighted_critics_sum += rating_critics_avg * weight
                        total_weight += weight
                        
                        print(f"Rating statistics - Audience avg: {rating_audience_avg:.1f}%, Critics avg: {rating_critics_avg:.1f}%")
                
                # Try to get year statistics as well
                if 'Year' in movie_info and weighted_scores and 'ReleaseYearGroup' in weighted_scores:
                    year = self.safe_convert_to_numeric(movie_info['Year'])
                    # Find decade or closest year group
                    for year_group in sorted(weighted_scores['ReleaseYearGroup'].keys()):
                        year_start, year_end = year_group.split('-')
                        if int(year_start) <= year <= int(year_end):
                            year_data = weighted_scores['ReleaseYearGroup'][year_group]
                            year_aud = year_data.get('aud', 0)
                            year_crit = year_data.get('crit', 0)
                            
                            # Convert if in 0-1 range
                            if year_aud <= 1.0 and year_crit <= 1.0:
                                year_aud *= 100
                                year_crit *= 100
                            
                            # Add to weighted sum (lower weight for year period)
                            weight = 1.0
                            stats_sources.append(f"Time period (weight: {weight:.1f})")
                            weighted_audience_sum += year_aud * weight
                            weighted_critics_sum += year_crit * weight
                            total_weight += weight
                            
                            print(f"Time period statistics - Audience avg: {year_aud:.1f}%, Critics avg: {year_crit:.1f}%")
                            break
                
                # Calculate weighted average from all sources
                stats_unrealistic = False
                if total_weight > 0:
                    weighted_audience_avg = weighted_audience_sum / total_weight
                    weighted_critics_avg = weighted_critics_sum / total_weight
                    
                    # Limit to reasonable range
                    weighted_audience_avg = min(95, max(30, weighted_audience_avg))
                    weighted_critics_avg = min(95, max(30, weighted_critics_avg))
                    
                    # Store weighted averages for potential fallback use
                    self._last_weighted_audience = weighted_audience_avg
                    self._last_weighted_critics = weighted_critics_avg
                    self._last_weighted_sources = stats_sources
                    
                    # Calculate reasonable range for predictions
                    max_realistic_audience = min(100, weighted_audience_avg + 15)
                    min_realistic_audience = max(0, weighted_audience_avg - 15)
                    max_realistic_critics = min(100, weighted_critics_avg + 15)
                    min_realistic_critics = max(0, weighted_critics_avg - 15)
                    
                    print(f"\nWeighted average from {len(stats_sources)} sources: {', '.join(stats_sources)}")
                    print(f"Weighted average - Audience: {weighted_audience_avg:.1f}%, Critics: {weighted_critics_avg:.1f}%")
                    print(f"Realistic ranges - Audience: {min_realistic_audience:.1f}-{max_realistic_audience:.1f}%, Critics: {min_realistic_critics:.1f}-{max_realistic_critics:.1f}%")
                    
                    # Check if predictions are outside statistical ranges
                    if (audience_prediction > max_realistic_audience or 
                        audience_prediction < min_realistic_audience or
                        critics_prediction > max_realistic_critics or
                        critics_prediction < min_realistic_critics):
                        print("WARNING: Model predictions are outside weighted statistical realistic ranges")
                        stats_unrealistic = True
                
                # If model isn't giving reasonable predictions, use statistics-based approach
                if model_seems_broken or stats_unrealistic:
                    print("Using statistics-based prediction instead of model prediction")
                    return self.fallback_prediction(movie_info)
                
                # Removed random offset - using pure model predictions
                
                # Ensure valid score ranges
                audience_prediction = max(0, min(100, audience_prediction))
                critics_prediction = max(0, min(100, critics_prediction))
                
                # Always ensure scores are in percentage format (0-100)
                if audience_prediction <= 1.0:
                    audience_prediction *= 100
                if critics_prediction <= 1.0:
                    critics_prediction *= 100
                
                # Generate explanation
                explanation = self.explain_prediction(movie_info, audience_prediction, critics_prediction)
                
                # Round scores to 2 decimal places
                audience_score = round(audience_prediction, 2)
                critics_score = round(critics_prediction, 2)
                
                print(f"Final scores: Audience {audience_score:.2f}%, Critics {critics_score:.2f}%")
                    
                return {
                    "title": movie_info.get('Title', 'Untitled Movie'),
                    "audience_score": audience_score,
                    "critics_score": critics_score,
                    "explanation": explanation
                }
                
            except Exception as e:
                print(f"Error processing features: {e}")
                traceback.print_exc()
                return self.fallback_prediction(movie_info)
                
        except Exception as e:
            print(f"Error predicting scores: {e}")
            traceback.print_exc()
            return self.fallback_prediction(movie_info)
    
    def fallback_prediction(self, movie_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide a fallback prediction when model prediction fails
        
        Args:
            movie_info (dict): Dictionary containing movie information
            
        Returns:
            dict: Dictionary with basic predictions
        """
        print("Using statistics-based prediction instead of model prediction")
        
        # Kiểm tra xem đã có weighted average từ phương thức predict chưa
        if hasattr(self, '_last_weighted_audience') and hasattr(self, '_last_weighted_critics'):
            audience_score = self._last_weighted_audience
            critics_score = self._last_weighted_critics
            
            print("Using previously calculated weighted average")
            print(f"Weighted average from {len(self._last_weighted_sources)} sources: {', '.join(self._last_weighted_sources)}")
            print(f"Final statistics-based prediction: Audience {audience_score:.1f}%, Critics {critics_score:.1f}%")
            
            # Generate explanation
            explanation = self.explain_prediction(movie_info, audience_score, critics_score)
            
            return {
                "title": movie_info.get('Title', 'Untitled Movie'),
                "audience_score": round(audience_score, 2),
                "critics_score": round(critics_score, 2),
                "explanation": explanation
            }
        
        # Nếu không có sẵn weighted average, tính toán lại
        print("Computing fresh weighted average from historical statistics...")
        
        # Generate more varied fallback predictions based on genre and rating
        audience_score = 70.0
        critics_score = 65.0
        
        # Adjust based on genres if available
        selected_genres = []
        for genre in ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
                      'Documentary', 'Drama', 'Family', 'Fantasy', 'Horror',
                      'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'Western']:
            if movie_info.get(genre, 0) == 1:
                selected_genres.append(genre)
        
        # Get weighted scores from model_info
        weighted_scores = self.model_info.get('weighted_scores', {}) if isinstance(self.model_info, dict) else {}
        
        # Calculate scores based on weighted historical data if available
        if weighted_scores:
            # Initialize aggregation variables
            weighted_audience_sum = 0
            weighted_critics_sum = 0
            total_weight = 0

            # Add genre statistics with high weight
            genre_audience_scores = []
            genre_critics_scores = []
            genre_weights = []
            genre_stats = []
            
            # First check genre_metrics for more detailed stats
            if 'genre_metrics' in weighted_scores:
                for genre in selected_genres:
                    if genre in weighted_scores['genre_metrics']:
                        metrics = weighted_scores['genre_metrics'][genre]
                        count = metrics.get('count', 0)
                        
                        if count > 0:
                            # Get mean scores
                            aud_mean = metrics.get('aud_mean', 0)
                            crit_mean = metrics.get('crit_mean', 0)
                            
                            # Convert if in 0-1 range
                            if aud_mean <= 1.0 and crit_mean <= 1.0:
                                aud_mean *= 100
                                crit_mean *= 100
                            
                            genre_audience_scores.append(aud_mean)
                            genre_critics_scores.append(crit_mean)
                            genre_weights.append(count)  # Weight by count
                            
                            # Store stats for explanation
                            genre_stats.append({
                                'genre': genre,
                                'count': count,
                                'aud_mean': aud_mean,
                                'crit_mean': crit_mean,
                                'aud_std': metrics.get('aud_std', 0) * 100 if metrics.get('aud_std', 0) <= 1.0 else metrics.get('aud_std', 0),
                                'crit_std': metrics.get('crit_std', 0) * 100 if metrics.get('crit_std', 0) <= 1.0 else metrics.get('crit_std', 0),
                                'aud_conf_int': metrics.get('aud_conf_int', (0, 0)),
                                'crit_conf_int': metrics.get('crit_conf_int', (0, 0))
                            })
            
            # Then check simple genre data if metrics not available
            elif 'Genre' in weighted_scores:
                for genre in selected_genres:
                    if genre in weighted_scores['Genre']:
                        genre_data = weighted_scores['Genre'][genre]
                        
                        # Get audience and critic scores
                        aud = genre_data.get('aud', 0)
                        crit = genre_data.get('crit', 0)
                        
                        # Convert if in 0-1 range
                        if aud <= 1.0 and crit <= 1.0:
                            aud *= 100
                            crit *= 100
                        
                        genre_audience_scores.append(aud)
                        genre_critics_scores.append(crit)
                        
                        # Weight by number of reviews
                        weight = max(1, (genre_data.get('aud_count', 0) + genre_data.get('crit_count', 0)) / 2)
                        genre_weights.append(weight)
                        
                        # Store stats for explanation
                        genre_stats.append({
                            'genre': genre,
                            'count': genre_data.get('movie_count', 0),
                            'aud_mean': aud,
                            'crit_mean': crit,
                            'aud_count': genre_data.get('aud_count', 0),
                            'crit_count': genre_data.get('crit_count', 0)
                        })
            
            # Calculate weighted genre averages if we have data
            if genre_audience_scores and sum(genre_weights) > 0:
                audience_from_genres = sum(a * w for a, w in zip(genre_audience_scores, genre_weights)) / sum(genre_weights)
                critics_from_genres = sum(c * w for c, w in zip(genre_critics_scores, genre_weights)) / sum(genre_weights)
                print(f"Genre-based averages: Audience {audience_from_genres:.1f}%, Critics {critics_from_genres:.1f}%")
                
                # Update base scores with genre-based values
                audience_score = audience_from_genres
                critics_score = critics_from_genres
                
                # Add to weighted sum (high weight for genre)
                weight = 3.0 * len(selected_genres)
                weighted_audience_sum += audience_from_genres * weight
                weighted_critics_sum += critics_from_genres * weight
                total_weight += weight
            
            # Extract director info
            director_stats = None
            if 'Director' in movie_info and movie_info['Director'] and 'Director' in weighted_scores:
                director = movie_info['Director']
                if director in weighted_scores['Director']:
                    director_data = weighted_scores['Director'][director]
                    
                    # Get audience and critic scores
                    aud = director_data.get('aud', 0)
                    crit = director_data.get('crit', 0)
                    
                    # Convert if in 0-1 range
                    if aud <= 1.0 and crit <= 1.0:
                        aud *= 100
                        crit *= 100
                    
                    # Store for explanation
                    director_stats = {
                        'director': director,
                        'aud_mean': aud,
                        'crit_mean': crit,
                        'aud_count': director_data.get('aud_count', 0),
                        'crit_count': director_data.get('crit_count', 0),
                        'movie_count': director_data.get('movie_count', 0),
                        'movies': director_data.get('movies', [])
                    }
                    
                    # Add to weighted sum (medium-high weight for director)
                    weight = 2.5
                    weighted_audience_sum += aud * weight
                    weighted_critics_sum += crit * weight
                    total_weight += weight
            
            # Adjust for characters
            character_audience_scores = []
            character_critics_scores = []
            character_weights = []
            character_stats = []
            
            # Extract characters
            selected_characters = []
            for i in range(1, 11):  # Check up to 10 character fields
                char_key = f'Character{i}'
                if char_key in movie_info and movie_info[char_key]:
                    selected_characters.append(movie_info[char_key])
            
            if 'Character' in weighted_scores:
                for character in selected_characters:
                    if character in weighted_scores['Character']:
                        char_data = weighted_scores['Character'][character]
                        
                        # Get audience and critic scores
                        aud = char_data.get('aud', 0)
                        crit = char_data.get('crit', 0)
                        
                        # Convert if in 0-1 range
                        if aud <= 1.0 and crit <= 1.0:
                            aud *= 100
                            crit *= 100
                        
                        character_audience_scores.append(aud)
                        character_critics_scores.append(crit)
                        
                        # Weight by number of reviews
                        weight = max(1, (char_data.get('aud_count', 0) + char_data.get('crit_count', 0)) / 2)
                        character_weights.append(weight)
                        
                        # Store for explanation
                        character_stats.append({
                            'character': character,
                            'aud_mean': aud,
                            'crit_mean': crit,
                            'aud_count': char_data.get('aud_count', 0),
                            'crit_count': char_data.get('crit_count', 0),
                            'movie_count': char_data.get('movie_count', 0),
                            'movies': char_data.get('movies', [])
                        })
            
            # Adjust with character data if available
            if character_audience_scores and sum(character_weights) > 0:
                character_audience = sum(a * w for a, w in zip(character_audience_scores, character_weights)) / sum(character_weights)
                character_critics = sum(c * w for c, w in zip(character_critics_scores, character_weights)) / sum(character_weights)
                
                # Add to weighted sum (medium weight for characters)
                weight = 2.0 * min(3, len(selected_characters))
                weighted_audience_sum += character_audience * weight
                weighted_critics_sum += character_critics * weight
                total_weight += weight
            
            # Rating statistics
            rating_stats = None
            if 'Rating' in movie_info and movie_info['Rating'] and 'Rating' in weighted_scores:
                rating = movie_info['Rating']
                if rating in weighted_scores['Rating']:
                    rating_data = weighted_scores['Rating'][rating]
                    
                    # Get audience and critic scores
                    rating_aud = rating_data.get('aud', 0)
                    rating_crit = rating_data.get('crit', 0)
                    
                    # Convert if in 0-1 range
                    if rating_aud <= 1.0 and rating_crit <= 1.0:
                        rating_aud *= 100
                        rating_crit *= 100
                    
                    # Store for explanation
                    rating_stats = {
                        'rating': rating,
                        'aud_mean': rating_aud,
                        'crit_mean': rating_crit,
                        'aud_count': rating_data.get('aud_count', 0),
                        'crit_count': rating_data.get('crit_count', 0),
                        'movie_count': rating_data.get('movie_count', 0)
                    }
                    
                    # Add to weighted sum (lower weight for rating)
                    weight = 1.5
                    weighted_audience_sum += rating_aud * weight
                    weighted_critics_sum += rating_crit * weight
                    total_weight += weight
            
            # Get year/time period statistics
            year_stats = None
            if 'Year' in movie_info and weighted_scores and 'ReleaseYearGroup' in weighted_scores:
                year = self.safe_convert_to_numeric(movie_info['Year'])
                # Find decade or closest year group
                for year_group in sorted(weighted_scores['ReleaseYearGroup'].keys()):
                    year_start, year_end = year_group.split('-')
                    if int(year_start) <= year <= int(year_end):
                        year_data = weighted_scores['ReleaseYearGroup'][year_group]
                        
                        # Get audience and critic scores
                        year_aud = year_data.get('aud', 0)
                        year_crit = year_data.get('crit', 0)
                        
                        # Convert if in 0-1 range
                        if year_aud <= 1.0 and year_crit <= 1.0:
                            year_aud *= 100
                            year_crit *= 100
                        
                        # Store for explanation
                        year_stats = {
                            'year_group': year_group,
                            'aud_mean': year_aud,
                            'crit_mean': year_crit,
                            'movie_count': year_data.get('movie_count', 0),
                            'aud_count': year_data.get('aud_count', 0),
                            'crit_count': year_data.get('crit_count', 0)
                        }
                        
                        # Add to weighted sum (low weight for year)
                        weight = 1.0
                        weighted_audience_sum += year_aud * weight
                        weighted_critics_sum += year_crit * weight
                        total_weight += weight
                        break
        
            # Calculate final weighted average
            if total_weight > 0:
                audience_score = weighted_audience_sum / total_weight
                critics_score = weighted_critics_sum / total_weight
                
                # Apply reasonable limits
                audience_score = min(95, max(30, audience_score))
                critics_score = min(95, max(30, critics_score))
                
                print(f"Using pure weighted average: Audience {audience_score:.1f}%, Critics {critics_score:.1f}%")
            else:
                # Default values if no weighted data available
                audience_score = 70.0
                critics_score = 65.0
                print("No weighted data available, using default scores")
                
                # Skip all other adjustments and directly use the weighted average
        
        # Ensure valid score ranges
        audience_score = max(0, min(100, audience_score))
        critics_score = max(0, min(100, critics_score))
        
        print(f"Final statistics-based prediction: Audience {audience_score:.1f}%, Critics {critics_score:.1f}%")
        
        # Generate a detailed explanation
        # Start with basic prediction information
        explanation = []
        explanation.append(f"Predicted audience score: {audience_score:.2f}%")
        explanation.append(f"Predicted critics score: {critics_score:.2f}%")
        
        # Add consensus analysis
        score_diff = abs(audience_score - critics_score)
        if score_diff < 10:
            explanation.append("\nStrong consensus between critics and audiences")
        elif score_diff < 20:
            explanation.append("\nModerate agreement between critics and audiences")
        else:
            if audience_score > critics_score:
                explanation.append("\nAudiences likely to enjoy this more than critics")
                # Add reasons if available from stats
                if weighted_scores and genre_stats:
                    reasons = []
                    for stat in genre_stats:
                        if stat.get('aud_mean', 0) > stat.get('crit_mean', 0) + 5:
                            reasons.append(f"{stat['genre']} films tend to be preferred by audiences")
                    if reasons:
                        explanation.append("Possible reasons:")
                        for reason in reasons[:3]:
                            explanation.append(f"- {reason}")
            else:
                explanation.append("\nCritics likely to rate this higher than general audiences")
                # Add reasons if available from stats
                if weighted_scores and genre_stats:
                    reasons = []
                    for stat in genre_stats:
                        if stat.get('crit_mean', 0) > stat.get('aud_mean', 0) + 5:
                            reasons.append(f"{stat['genre']} films tend to be preferred by critics")
                    if reasons:
                        explanation.append("Possible reasons:")
                        for reason in reasons[:3]:
                            explanation.append(f"- {reason}")
        
        # Add statistical breakdown section
        explanation.append("\n=== Statistical Score Breakdown ===")
        
        # Add model info if available
        if isinstance(self.model_info, dict) and 'feature_importance' in self.model_info:
            feature_importance = self.model_info['feature_importance']
            explanation.append("\nTop predictive features for this type of movie:")
            
            # Get audience feature importance
            if 'audience' in feature_importance:
                audience_features = feature_importance['audience']
                top_audience = sorted(audience_features.items(), key=lambda x: x[1], reverse=True)[:5]
                explanation.append("Audience score prediction influenced by:")
                for i, (feature, importance) in enumerate(top_audience):
                    # Make feature name more readable
                    readable_feature = f"Feature {i+1}"
                    # Limit percentage to reasonable values
                    importance_pct = min(100, importance*100)
                    explanation.append(f"  - {readable_feature}: {importance_pct:.1f}% importance")
            
            # Get critics feature importance
            if 'critics' in feature_importance:
                critics_features = feature_importance['critics']
                top_critics = sorted(critics_features.items(), key=lambda x: x[1], reverse=True)[:5]
                explanation.append("\nCritics score prediction influenced by:")
                for i, (feature, importance) in enumerate(top_critics):
                    # Make feature name more readable
                    readable_feature = f"Feature {i+1}"
                    # Limit percentage to reasonable values
                    importance_pct = min(100, importance*100)
                    explanation.append(f"  - {readable_feature}: {importance_pct:.1f}% importance")
        
        # Add director information if available
        if director_stats:
            explanation.append(f"\nDirector Analysis ({director_stats['director']}):")
            explanation.append(f"- Previous films averaged {director_stats['aud_mean']:.1f}% audience, {director_stats['crit_mean']:.1f}% critics")
            explanation.append(f"- Based on {director_stats['aud_count']:.0f} audience reviews and {director_stats['crit_count']:.0f} critic reviews")
            
            # Add film list if available
            if 'movies' in director_stats and director_stats['movies']:
                explanation.append(f"- Notable films by {director_stats['director']}:")
                for i, movie in enumerate(director_stats['movies']):
                    if i >= 3:  # Limit to 3 movies
                        break
                    title = movie.get('title', 'Unknown')
                    a_score = movie.get('audience_score')
                    c_score = movie.get('critics_score')
                    if a_score is not None and c_score is not None:
                        # Ensure scores are in percentage format
                        if a_score <= 1.0:
                            a_score *= 100
                        if c_score <= 1.0:
                            c_score *= 100
                        explanation.append(f"  * {title} (Audience: {a_score:.1f}%, Critics: {c_score:.1f}%)")
                    else:
                        explanation.append(f"  * {title}")
        
        # Add character information if available
        for i, char_stat in enumerate(character_stats):
            if i >= 3:  # Limit to 3 characters
                break
            explanation.append(f"\nCharacter Analysis ({char_stat['character']}):")
            explanation.append(f"- Films with this character averaged {char_stat['aud_mean']:.1f}% audience, {char_stat['crit_mean']:.1f}% critics")
            explanation.append(f"- Based on {char_stat['aud_count']:.0f} audience reviews and {char_stat['crit_count']:.0f} critic reviews")
        
        # Add rating information if available
        if rating_stats:
            explanation.append(f"\nRating Analysis ({rating_stats['rating']}):")
            explanation.append(f"- Films with this rating averaged {rating_stats['aud_mean']:.1f}% audience, {rating_stats['crit_mean']:.1f}% critics")
            explanation.append(f"- Based on {rating_stats['movie_count']:.0f} movies")
        
        # Add genre analysis
        if selected_genres:
            explanation.append(f"\nGenre Analysis:")
            explanation.append(f"- Selected genres: {', '.join(selected_genres)}")
            
            # Add detailed genre stats
            if genre_stats:
                explanation.append("\nGenre Historical Statistics:")
                for stat in genre_stats:
                    explanation.append(f"- {stat['genre']} films ({stat.get('count', 0)} analyzed):")
                    explanation.append(f"  * Audience: {stat['aud_mean']:.1f}% (±{stat.get('aud_std', 0):.1f}%)")
                    explanation.append(f"  * Critics: {stat['crit_mean']:.1f}% (±{stat.get('crit_std', 0):.1f}%)")
                    
                    # Add confidence intervals if available
                    if 'aud_conf_int' in stat and 'crit_conf_int' in stat:
                        aud_low, aud_high = stat['aud_conf_int']
                        crit_low, crit_high = stat['crit_conf_int']
                        
                        # Convert to percentages if in decimal form
                        if aud_low <= 1.0 and aud_high <= 1.0:
                            aud_low *= 100
                            aud_high *= 100
                            crit_low *= 100
                            crit_high *= 100
                        
                        # Apply limits
                        aud_low = max(0, min(100, aud_low))
                        aud_high = min(100, aud_high)
                        crit_low = max(0, min(100, crit_low))
                        crit_high = min(100, crit_high)
                        
                        explanation.append(f"  * 95% confidence: Audience scores between {aud_low:.1f}% and {aud_high:.1f}%")
                        explanation.append(f"  * 95% confidence: Critic scores between {crit_low:.1f}% and {crit_high:.1f}%")
        
        # Add time period analysis
        if year_stats:
            explanation.append(f"\nTime Period Analysis:")
            explanation.append(f"- Films from {year_stats['year_group']} averaged {year_stats['aud_mean']:.1f}% audience, {year_stats['crit_mean']:.1f}% critics")
            explanation.append(f"- Based on {year_stats['movie_count']} movies from this time period")
        
        # Add model performance metrics if available
        if isinstance(self.model_info, dict) and 'model_metrics' in self.model_info:
            metrics = self.model_info['model_metrics']
            explanation.append("\nModel Performance Statistics:")
            
            if 'audience_r2' in metrics:
                audience_r2 = metrics['audience_r2']
                audience_rmse = metrics.get('audience_rmse', 0)
                
                # Convert to percentage if in decimal form
                if audience_r2 <= 1.0:
                    audience_r2 = audience_r2 * 100
                if audience_rmse <= 1.0:
                    audience_rmse = audience_rmse * 100
                
                explanation.append(f"- Audience score prediction accuracy: {audience_r2:.1f}%")
                explanation.append(f"- Average audience score error: ±{audience_rmse:.1f}%")
            
            if 'critics_r2' in metrics:
                critics_r2 = metrics['critics_r2']
                critics_rmse = metrics.get('critics_rmse', 0)
                
                # Convert to percentage if in decimal form
                if critics_r2 <= 1.0:
                    critics_r2 = critics_r2 * 100
                if critics_rmse <= 1.0:
                    critics_rmse = critics_rmse * 100
                
                explanation.append(f"- Critics score prediction accuracy: {critics_r2:.1f}%")
                explanation.append(f"- Average critics score error: ±{critics_rmse:.1f}%")
        
        # Join all explanation parts
        complete_explanation = '\n'.join(explanation)
        
        return {
            "title": movie_info.get('Title', 'Untitled Movie'),
            "audience_score": round(audience_score, 2),
            "critics_score": round(critics_score, 2),
            "explanation": complete_explanation
        } 