import pandas as pd
import numpy as np
import asyncio
import aiohttp
import json
import re
import os
import sys
import pickle
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import functions from utils.model_trainer
from .utils.model_trainer import (
    train_models, 
    compute_weighted_scores,
    analyze_movie_data,
    analyze_genre_trends,
    analyze_rating_discrepancies,
    perform_genre_clustering
)

class ApiDataTrainer:
    def __init__(self, api_base_url="http://localhost:8000"):
        """Initialize the trainer with API base URL"""
        self.api_base_url = api_base_url
        self.all_movies = []
        
    async def fetch_data_from_api(self, limit=20, max_pages=100):
        """Fetch movie data from the API endpoint"""
        logger.info(f"Fetching data from API at {self.api_base_url}")
        
        # First check if API is working with a simple request
        async with aiohttp.ClientSession() as session:
            try:
                check_url = f"{self.api_base_url}/api/movies/?limit=1"
                logger.info(f"Testing API connection with: {check_url}")
                
                async with session.get(check_url) as response:
                    if response.status != 200:
                        logger.error(f"API not available: HTTP {response.status}")
                        return []
                    
                    data = await response.json()
                    if not data.get("movies"):
                        logger.error("API returned no movies in test request")
                        return []
                    
                    logger.info("API connection test successful")
            except Exception as e:
                logger.error(f"Error connecting to API: {str(e)}")
                return []
        
        all_movies = []
        next_token = None
        page_count = 0
        
        async with aiohttp.ClientSession() as session:
            while page_count < max_pages:
                # Construct URL with pagination
                url = f"{self.api_base_url}/api/movies/?limit={limit}"
                if next_token:
                    url += f"&pagination_token={next_token}"
                
                try:
                    logger.info(f"Fetching page {page_count+1}...")
                    async with session.get(url) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Error fetching data: HTTP {response.status}")
                            logger.error(f"Response body: {error_text[:200]}...")  # Print first 200 chars
                            break
                        
                        data = await response.json()
                        movies = data.get("movies", [])
                        
                        if not movies:
                            logger.info("No more movies found")
                            break
                        
                        all_movies.extend(movies)
                        logger.info(f"Fetched page {page_count+1} with {len(movies)} movies (Total: {len(all_movies)})")
                        
                        next_token = data.get("pagination_token")
                        if not next_token:
                            logger.info("No more pages available")
                            break
                        
                        page_count += 1
                        
                except Exception as e:
                    logger.error(f"Error fetching data: {str(e)}")
                    break
        
        self.all_movies = all_movies
        logger.info(f"Total movies fetched: {len(all_movies)}")
        return all_movies
    
    def filter_movies(self):
        """Filter out 2025 movies with no ratings"""
        if not self.all_movies:
            return []
        
        filtered_movies = []
        movies_filtered = 0
        
        for movie in self.all_movies:
            metadata = movie.get("metadata", {})
            
            # Extract release year from release date or metadataArr
            release_year = None
            release_date = metadata.get("releaseDate", "")
            if not release_date:
                # Try to find it in metadataArr
                for meta_item in metadata.get("metadataArr", []):
                    if isinstance(meta_item, str) and meta_item.startswith("Released"):
                        release_date = meta_item
                        break
            
            # Extract year using regex
            if isinstance(release_date, str):
                year_match = re.search(r'\b(19|20)\d{2}\b', release_date)
                if year_match:
                    release_year = int(year_match.group(0))
            
            # Get scores
            audience_score = metadata.get("audienceScore", "")
            critics_score = metadata.get("criticsScore", "")
            
            has_audience_score = audience_score and audience_score != "0%" and audience_score != "0"
            has_critics_score = critics_score and critics_score != "0%" and critics_score != "0"
            
            # Filter out 2025 movies with no ratings
            if release_year == 2025 and not (has_audience_score or has_critics_score):
                movies_filtered += 1
                continue
            
            filtered_movies.append(movie)
        
        # Replace the all_movies with filtered list
        self.all_movies = filtered_movies
        logger.info(f"Filtered out {movies_filtered} movies from 2025 with no ratings.")
        logger.info(f"Remaining movies: {len(filtered_movies)}")
        return filtered_movies
    
    def _ensure_percentage_format(self, value):
        """
        Ensure value is in 0-100 percentage range (consistent with original training)
        This is CRITICAL to match the original training file approach
        """
        if value is None:
            return None
            
        if isinstance(value, str):
            # Remove % if present
            value = value.replace('%', '')
            try:
                value = float(value)
            except ValueError:
                return None
        
        # Convert to 0-100 scale if in 0-1 range
        if isinstance(value, (int, float)) and 0 <= value <= 1:
            value = value * 100
            
        return value
    
    def convert_to_dataframe(self):
        """Convert API data to DataFrame format expected by movie_analysis.py"""
        if not self.all_movies:
            logger.error("No movie data available")
            return None
        
        # First filter the movies
        self.filter_movies()
        
        logger.info("Converting API data to DataFrame...")
        movie_data = []
        
        for movie in self.all_movies:
            movie_id = movie.get("id", "")
            metadata = movie.get("metadata", {})
            
            # Extract fields from metadata
            title = metadata.get("title", "")
            description = metadata.get("description", "")
            
            # Extract scores and ensure they're in percentage format (0-100 scale)
            audience_score = self._ensure_percentage_format(metadata.get("audienceScore", ""))
            critics_score = self._ensure_percentage_format(metadata.get("criticsScore", ""))
            
            logger.debug(f"Movie: {title}, Original scores: {metadata.get('audienceScore')} / {metadata.get('criticsScore')}")
            logger.debug(f"Converted scores: {audience_score} / {critics_score}")
                
            # Extract genres
            genres = metadata.get("genres", [])
            if isinstance(genres, str):
                # If genres is a comma-separated string, split it
                genres = [g.strip() for g in genres.split(",")]
            
            # Create genre dictionary with boolean flags for common genres
            common_genres = [
                "Action", "Adventure", "Animation", "Comedy", "Crime", 
                "Documentary", "Drama", "Family", "Fantasy", "Horror",
                "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "Western"
            ]
            
            genre_dict = {genre: 1 for genre in common_genres if genre in genres}
            
            # Create individual genre columns
            genre_cols = {}
            for i, genre in enumerate(genres[:5], 1):  # Up to 5 genres
                genre_cols[f"Genre{i}"] = genre
            
            # Fill remaining genre columns with empty strings
            for i in range(len(genres) + 1, 6):
                genre_cols[f"Genre{i}"] = ""
            
            # Handle combined genres
            genres_str = ", ".join(genres) if genres else ""
            
            # Extract metadata
            metadata_arr = metadata.get("metadataArr", [])
            
            # Extract rating, release date, and duration
            rating = ""
            release_date = ""
            duration = ""
            release_year = None
            
            for meta_item in metadata_arr:
                if isinstance(meta_item, str):
                    if re.match(r'^(G|PG|PG-13|R|NC-17)$', meta_item):
                        rating = meta_item
                    elif meta_item.startswith("Released"):
                        release_date = meta_item.replace("Released ", "")
                        # Try to extract year
                        year_match = re.search(r'\b(19|20)\d{2}\b', release_date)
                        if year_match:
                            release_year = int(year_match.group(0))
                    elif "h" in meta_item or "m" in meta_item:
                        duration = meta_item
            
            # Extract cast and director
            cast_str = metadata.get("cast", "")
            director = metadata.get("director", "")
            
            # If director is not explicitly provided, assume it's the first person in cast
            characters = []
            if cast_str and isinstance(cast_str, str):
                cast_list = [item.strip() for item in cast_str.split(",")]
                
                # If no director, use first cast member as director
                if not director and cast_list:
                    director = cast_list[0]
                    # Other cast members are characters (skip the first if it's used as director)
                    characters = cast_list[1:5] if not director else cast_list[:5]
                else:
                    characters = cast_list[:5]
            
            # Convert duration to minutes
            duration_minutes = None
            if duration:
                hours_match = re.search(r'(\d+)\s*h', duration)
                mins_match = re.search(r'(\d+)\s*m', duration)
                hours = int(hours_match.group(1)) if hours_match else 0
                mins = int(mins_match.group(1)) if mins_match else 0
                duration_minutes = hours * 60 + mins
            
            # Extract audience/critic counts and ensure they're numeric
            audience_verified_count = metadata.get("audienceVerifiedCount", "")
            critic_reviews = metadata.get("criticReviews", "")
            
            # Clean up counts (remove commas and "reviews" text)
            if isinstance(audience_verified_count, str):
                audience_verified_count = re.sub(r'[^\d]', '', audience_verified_count)
            if isinstance(critic_reviews, str):
                critic_reviews = re.sub(r'[^\d]', '', critic_reviews)
                
            # Convert to integers if possible
            try:
                audience_verified_count = int(audience_verified_count) if audience_verified_count else None
            except ValueError:
                audience_verified_count = None
                
            try:
                critic_reviews = int(critic_reviews) if critic_reviews else None
            except ValueError:
                critic_reviews = None
            
            # Extract consensus
            critics_consensus = metadata.get("criticsConsensus", "")
            audience_consensus = metadata.get("audienceConsensus", "")
            
            # Create data row
            movie_row = {
                "ID": movie_id,
                "Title": title,
                "Description": description,
                "AudienceScore": audience_score,  # Already in 0-100 format
                "CriticsScore": critics_score,    # Already in 0-100 format
                "Genres": genres_str,
                "Rating": rating,
                "ReleaseDate": release_date,
                "ReleaseYear": release_year,
                "Duration": duration,
                "Duration_Minutes": duration_minutes,
                "Director": director,
                "AudienceVerifiedCount": audience_verified_count,
                "CriticReviews": critic_reviews,
                "CriticsConsensus": critics_consensus,
                "AudienceConsensus": audience_consensus
            }
            
            # Add genre columns
            movie_row.update(genre_cols)
            
            # Add individual genre flags
            movie_row.update(genre_dict)
            
            # Add character columns
            for i, character in enumerate(characters[:5], 1):
                movie_row[f"Character{i}"] = character
            
            # Fill in missing character columns with empty strings
            for i in range(len(characters) + 1, 6):
                movie_row[f"Character{i}"] = ""
            
            # Calculate score difference (ensure both scores exist)
            if audience_score is not None and critics_score is not None:
                movie_row["ScoreDifference"] = audience_score - critics_score
                movie_row["ScoreDifferenceAbs"] = abs(movie_row["ScoreDifference"])
            
            movie_data.append(movie_row)
        
        # Create DataFrame
        df = pd.DataFrame(movie_data)
        
        logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        
        # Let's check the score ranges to ensure they're in the correct format
        if 'AudienceScore' in df.columns and len(df) > 0:
            aud_min = df['AudienceScore'].min() if not df['AudienceScore'].isna().all() else None
            aud_max = df['AudienceScore'].max() if not df['AudienceScore'].isna().all() else None
            crit_min = df['CriticsScore'].min() if not df['CriticsScore'].isna().all() else None
            crit_max = df['CriticsScore'].max() if not df['CriticsScore'].isna().all() else None
            
            logger.info(f"Score ranges - Audience: {aud_min} to {aud_max}, Critics: {crit_min} to {crit_max}")
            
            # Final check to ensure all scores are in 0-100 range
            if aud_max is not None and aud_max <= 1.0:
                logger.warning("Audience scores appear to be in 0-1 range, converting to 0-100")
                df['AudienceScore'] = df['AudienceScore'] * 100
                
            if crit_max is not None and crit_max <= 1.0:
                logger.warning("Critics scores appear to be in 0-1 range, converting to 0-100")
                df['CriticsScore'] = df['CriticsScore'] * 100
        
        return df
    
    def analyze_api_data(self):
        """Perform comprehensive analysis of the API data"""
        df = self.convert_to_dataframe()
        if df is None or df.empty:
            logger.error("Error: No valid data for analysis")
            return None
        
        logger.info("Starting movie data analysis...")
        
        # Create output directories if they don't exist
        os.makedirs('../rt-data-science-api/app/models', exist_ok=True)
        os.makedirs('../rt-data-science-api/app/processed', exist_ok=True)
        
        # Run comprehensive analysis using the imported function
        analysis_results = analyze_movie_data(df)
        
        if analysis_results:
            logger.info("Analysis completed successfully!")
            # Save timestamp
            with open('../rt-data-science-api/app/models/analysis_timestamp.txt', 'w') as f:   
                f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return analysis_results
        else:
            logger.error("Analysis failed")
            return None
    
    def train_model_from_api_data(self):
        """Train the model using the converted DataFrame"""
        df = self.convert_to_dataframe()
        if df is None or df.empty:
            logger.error("Error: No valid data for model training")
            return None
        
        logger.info("Starting model training...")
        
        # Create required output directories (multiple paths to ensure compatibility)
        directories = [
            '../rt-data-science-api/app/models',  # Relative from scripts dir
            'app/models',                         # Direct app path
            '../models'                           # Original path
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Ensuring directory exists: {directory}")
        
        # Train the model using the imported function from movie_analysis.py
        model_info = train_models(data_file=df, model_output_file="api_trained_model")
        
        if model_info:
            logger.info("Model training completed successfully!")
            
            # Ensure models are saved in all required locations
            model_files = {
                "audience_score_model.pkl": model_info.get('audience_model', None),
                "critics_score_model.pkl": model_info.get('critics_model', None), 
                "model_info.pkl": model_info,
                "weighted_scores.pkl": model_info.get('weighted_scores', {})
            }
            
            # Save to multiple locations for compatibility
            for model_path in ['../rt-data-science-api/app/models', 'app/models', '../models']:
                logger.info(f"Saving models to {model_path}")
                
                for filename, data in model_files.items():
                    if data is not None:
                        try:
                            with open(f"{model_path}/{filename}", 'wb') as f:
                                pickle.dump(data, f)
                            logger.info(f"Saved {filename} to {model_path}")
                        except Exception as e:
                            logger.error(f"Error saving {filename} to {model_path}: {str(e)}")
            
            # Save timestamp
            with open('app/models/training_timestamp.txt', 'w') as f:
                training_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"Training completed: {training_time}")
                
            logger.info(f"All models saved successfully at: {training_time}")
            return model_info
        else:
            logger.error("Model training failed")
            return None

async def main():
    trainer = ApiDataTrainer()
    # Fetch movie data from API - use smaller limit to avoid 422 errors
    await trainer.fetch_data_from_api(limit=20, max_pages=100)
    
    # Choose which operation to perform
    operation = "all"  # Options: "train", "analyze", "all"
    
    if operation in ["train", "all"]:
        trainer.train_model_from_api_data()
    
    if operation in ["analyze", "all"]:
        trainer.analyze_api_data()

if __name__ == "__main__":
    asyncio.run(main()) 