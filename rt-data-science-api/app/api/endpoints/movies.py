from fastapi import APIRouter, HTTPException, Path, Body, Query
from typing import List, Dict, Optional, Any
from services.vector_service import VectorService
from core.config import RECORD_NAMESPACE
from models.pinecone_model import Movie, MovieResponse
from models.schemas import MoviePaginationRequest, MovieSearchRequest, MovieRecommendRequest, MoviePredictionRequest, MoviePredictionResponse
from models.movie_prediction import MoviePredictor
from utils.movie_text_processor import MovieTextProcessor
import re
import os
import os.path
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
vector_service = VectorService()

# Initialize the movie predictor (lazy-loaded when needed)
_movie_predictor = None

def get_movie_predictor():
    """Get or initialize the MoviePredictor singleton"""
    global _movie_predictor
    if _movie_predictor is None:
        try:
            # Try using absolute paths to models directory
            model_paths = [
                ('app/models/audience_score_model.pkl', 'app/models/critics_score_model.pkl', 'app/models/model_info.pkl'),
                ('../rt-data-science-api/app/models/audience_score_model.pkl', '../rt-data-science-api/app/models/critics_score_model.pkl', '../rt-data-science-api/app/models/model_info.pkl'),
                ('../models/audience_score_model.pkl', '../models/critics_score_model.pkl', '../models/model_info.pkl')
            ]
            
            for paths in model_paths:
                if all(os.path.exists(path) for path in paths):
                    _movie_predictor = MoviePredictor(
                        audience_model_path=paths[0],
                        critics_model_path=paths[1],
                        model_info_path=paths[2]
                    )
                    logger.info(f"Loaded movie predictor using paths: {paths}")
                    break
            
            # If no valid paths found, use default paths as fallback
            if _movie_predictor is None:
                logger.warning("No valid model paths found, using default paths")
                _movie_predictor = MoviePredictor(
                    audience_model_path='app/models/audience_score_model.pkl',
                    critics_model_path='app/models/critics_score_model.pkl',
                    model_info_path='app/models/model_info.pkl'
                )
        except Exception as e:
            logger.error(f"Error initializing predictor: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize prediction model: {str(e)}")
    return _movie_predictor

@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """
    Health check endpoint to verify if the service is running.
    """
    return {"status": "ok"}

@router.get("/", response_model=Dict[str, Any])
async def get_movies_by_page(
    limit: int = Query(10, ge=1, le=50, description="Number of movies to return per page"),
    pagination_token: Optional[str] = Query(None, description="Token for getting the next page")
):
    """
    Get paginated list of movies.
    
    Parameters:
    - limit: Number of movies to return per page
    - pagination_token: Token for getting the next page (null when there are no more pages)
    """
    try:
        # Get list of vector IDs
        result = await vector_service.list_vectors(
            namespace=RECORD_NAMESPACE,
            pagination_token=pagination_token,
            limit=limit
        )
        
        vector_ids = result.get("vector_ids", [])
        next_token = result.get("next_pagination_token")
        
        logger.info(f"Retrieved {len(vector_ids)} vector IDs")
        
        # Empty response is valid - return early
        if not vector_ids:
            return {
                "movies": [],
                "pagination_token": next_token
            }
        
        # Process movies in smaller batches to avoid large requests
        movies = []
        batch_size = 10
        
        for i in range(0, len(vector_ids), batch_size):
            batch_ids = vector_ids[i:i + batch_size]
            try:
                vectors = await vector_service.fetch_vectors(ids=batch_ids)
               
                # Process each movie in the batch
                for movie_id, vector_data in vectors.items():
                    try:
                        movie_data = vector_data.get("metadata", {})
                        values = vector_data.get("values", [])
                        
                        # Create a Movie object from metadata
                        movie = Movie(**movie_data)
                        
                        # Add to the movies list
                        movies.append({
                            "id": movie_id,
                            "metadata": movie
                        })
                    except Exception as e:
                        logger.error(f"Error processing movie {movie_id}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error fetching batch of vectors: {str(e)}")
                # Continue processing other batches even if one fails
        
        return {
            "movies": movies,
            "pagination_token": next_token
        }
        
    except Exception as e:
        logger.error(f"Error in get_movies_by_page: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch movies: {str(e)}")
    

@router.get("/{movie_id}", response_model=MovieResponse)
async def get_movie_by_id(movie_id: str = Path(..., description="The movie ID")):
    """
    Get a specific movie by its ID.
    """
    try:
        vectors = await vector_service.fetch_vectors(ids=[movie_id])
        
        if movie_id not in vectors:
            raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
        
        vector_data = vectors[movie_id]
        
        # Create a Movie object from metadata
        movie = Movie(**vector_data.get("metadata", {}))
        
        return MovieResponse(
            id=movie_id,
            values=vector_data.get("values", []),
            metadata=movie
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch movie: {str(e)}")

@router.post("/search", response_model=List[MovieResponse])
async def search_movies(request: MovieSearchRequest = Body(...)):
    """
    Search for movies by title and genres.
    """
    try:
        logger.info(f"Searching for: {request.query}")
        
        # Check for year in the query
        year_pattern = r'\b(19|20)\d{2}\b'
        year_match = re.search(year_pattern, request.query)
        year_filter = None
        if year_match:
            year_filter = year_match.group(0)
            logger.info(f"Detected year filter: {year_filter}")
        
        # Generate embedding for the query text
        embedding = await vector_service.get_text_embedding(request.query)
        
        if not embedding:
            raise HTTPException(status_code=400, detail="Failed to generate embedding for query")
        
        # Search for more movies than requested to allow for filtering
        search_top_k = request.top_k * 3 if year_filter else request.top_k
        results = await vector_service.search_vector(vector=embedding, top_k=search_top_k)
        
        movies = []
        for match in results.get("matches", []):
            try:
                # Extract metadata and handle missing fields gracefully
                movie_data = match.get("metadata", {})
                
                # Apply year filtering if requested
                if year_filter:
                    # Check various fields that might contain year information
                    movie_metadata_str = str(movie_data)
                    if year_filter not in movie_metadata_str:
                        continue
                
                # Check if any required fields are missing and add defaults
                required_fields = ["title", "description", "audienceConsensus", "cast", "criticsConsensus", "genres"]
                for field in required_fields:
                    if field not in movie_data:
                        movie_data[field] = ""
                        
                # Handle genres specifically - ensure it's a list
                if movie_data.get("genres") == "":
                    movie_data["genres"] = []
                
                # Create Movie object with relaxed validation
                movie = Movie.parse_obj(movie_data)
                
                movies.append(
                    MovieResponse(
                        id=match["id"],
                        score=match.get("score", 0.0),
                        values=match.get("values", [])[:10],  # Truncate values to reduce payload
                        metadata=movie
                    )
                )
                
                # If we have enough movies after filtering, stop
                if year_filter and len(movies) >= request.top_k:
                    break
                    
            except Exception as e:
                logger.error(f"Error processing search result: {str(e)}")
                continue
                
        return movies[:request.top_k]  # Ensure we don't return more than requested
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/recommend", response_model=List[MovieResponse])
async def get_recommendations(request: MovieRecommendRequest = Body(...)):
    """
    Get movie recommendations based on a text query.
    """
    try:
        # Process the query text with MovieTextProcessor
        # Create processor instance
        processor = MovieTextProcessor()

        # Process the query text to extract structured information
        processed_data = processor.process_movie_data(request.query)

        # Format the processed data for embedding
        query_for_embedding = processor.format_for_embedding(processed_data)

        # If the query is too generic or couldn't be processed well, use the original
        if not query_for_embedding or len(query_for_embedding) < 10:
            query_for_embedding = request.query

        logger.info(f"Original query: {request.query}")
        logger.info(f"Processed query for embedding: {query_for_embedding}")
        
        # Generate embedding for the query text
        embedding = await vector_service.get_text_embedding(query_for_embedding)
        
        if not embedding:
            raise HTTPException(status_code=400, detail="Failed to generate embedding for query")
        
        # Search for movies similar to the embedding
        results = await vector_service.search_vector(vector=embedding, top_k=request.top_k)
        
        movies = []
        for match in results.get("matches", []):
            try:
                # Extract metadata and handle missing fields gracefully
                movie_data = match.get("metadata", {})
                
                # Check if any required fields are missing and add defaults
                required_fields = ["title", "description", "audienceConsensus", "cast", "criticsConsensus"]
                for field in required_fields:
                    if field not in movie_data:
                        movie_data[field] = ""
                
                # Create Movie object with relaxed validation
                movie = Movie.parse_obj(movie_data)
                
                movies.append(
                    MovieResponse(
                        id=match["id"],
                        score=match.get("score", 0.0),
                        values=match.get("values", [])[:10],  # Truncate values
                        metadata=movie
                    )
                )
            except Exception as e:
                logger.error(f"Error processing recommendation: {str(e)}")
                continue
                
        return movies
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@router.post("/predict", response_model=MoviePredictionResponse)
async def predict_movie_scores(request: MoviePredictionRequest):
    """Predict audience and critics scores for a movie"""
    try:
        # Get the movie predictor
        movie_predictor = get_movie_predictor()
        
        # Convert request to dict
        movie_info = request.dict(exclude_unset=True)
        
        # Handle SciFi vs Sci-Fi mapping
        if 'SciFi' in movie_info and movie_info['SciFi'] == 1:
            movie_info['Sci-Fi'] = 1
        
        # Debug - log what we're passing to the model
        logger.info(f"Predicting for movie: {movie_info.get('Title')}")
        logger.info(f"Genres: {[g for g in ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'Western'] if movie_info.get(g, 0) == 1]}")
        
        # Make prediction
        logger.info("Calling movie_predictor.predict...")
        prediction = movie_predictor.predict(movie_info)
        
        # Ensure scores are in percentage format (0-100 scale)
        audience_score = prediction.get('audience_score', 0)
        critics_score = prediction.get('critics_score', 0)
        
        # Convert to percentage if in decimal format
        if audience_score <= 1.0:
            audience_score *= 100
        if critics_score <= 1.0:
            critics_score *= 100
            
        # Apply safe limits
        audience_score = max(0, min(100, audience_score))
        critics_score = max(0, min(100, critics_score))
        
        logger.info(f"Prediction successful - Audience: {audience_score:.2f}%, Critics: {critics_score:.2f}%")
        
        return {
            "title": prediction.get("title", movie_info.get("Title", "Unknown Movie")),
            "audience_score": round(audience_score, 2),
            "critics_score": round(critics_score, 2),
            "explanation": prediction.get("explanation", "No explanation available.")
        }
    except Exception as e:
        error_detail = str(e)
        logger.error(f"Error predicting movie scores: {error_detail}")
        
        if hasattr(e, "__traceback__"):
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            logger.error(f"Traceback: {tb_str}")
        
        # Try to use fallback directly
        try:
            logger.info("Attempting direct fallback prediction...")
            fallback = movie_predictor.fallback_prediction(request.dict())
            
            logger.info(f"Fallback successful - Audience: {fallback.get('audience_score', 0):.2f}%, Critics: {fallback.get('critics_score', 0):.2f}%")
            
            return {
                "title": fallback.get("title", request.Title),
                "audience_score": fallback.get("audience_score", 75.0),
                "critics_score": fallback.get("critics_score", 70.0),
                "explanation": fallback.get("explanation", f"Error with ML prediction: {error_detail}\n\nUsing statistical prediction instead.")
            }
        except Exception as fallback_error:
            logger.error(f"Even fallback prediction failed: {fallback_error}")
            
            # Return a simple fallback with the error
            return {
                "title": request.Title,
                "audience_score": 75.0,
                "critics_score": 70.0,
                "explanation": f"Error making prediction: {error_detail}\n\nFallback scores provided."
            }