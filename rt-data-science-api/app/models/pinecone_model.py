from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class Movie(BaseModel):
    """Movie data model representing movie details from Rotten Tomatoes or similar source."""
    
    title: str = ""
    description: str = ""
    audience_consensus: Optional[str] = Field(default="N/A", alias="audienceConsensus")
    audience_score: Optional[str] = Field(default="", alias="audienceScore")
    audience_verified_count: Optional[str] = Field(default="", alias="audienceVerifiedCount")
    cast: Optional[str] = ""
    critic_reviews: Optional[str] = Field(default="", alias="criticReviews")
    critics_consensus: Optional[str] = Field(default="", alias="criticsConsensus")
    critics_score: Optional[str] = Field(default="", alias="criticsScore")
    genres: Optional[str] = ""
    metadata: Optional[str] = ""
    visual: Optional[str] = None
    
    class Config:
        populate_by_name = True
        extra = "ignore"  # Ignore extra fields in the input data
        json_schema_extra = {
            "example": {
                "title": "Predators",
                "description": "Brought together on a mysterious planet, a mercenary (Adrien Brody) and a group of coldblooded killers now become the prey. A new breed of aliens pursues the ragtag humans through dense jungle. The group must work together to survive, or become the latest trophies of the fearsome intergalactic hunters.",
                "audienceConsensus": "N/A",
                "audienceScore": "52%",
                "audienceVerifiedCount": "100,000+ Ratings",
                "cast": "Nimród Antal, Adrien Brody, Topher Grace, Alice Braga, Walton Goggins, Oleg Taktarov, Laurence Fishburne, Danny Trejo, Louis Ozawa, Mahershala Ali, Alex Litvak, Michael Finch, Alex Young, Robert Rodriguez, John Davis, Elizabeth Avellan, Gyula Pados, Dan Zimmerman, John Debney, Steve Joyner, Caylah Eddleblute, Nina Proctor, Mary Vernieu, J.C. Cantu, David Hack",
                "criticReviews": "201 Reviews",
                "criticsConsensus": "After a string of subpar sequels, this bloody, action-packed reboot takes the Predator franchise back to its testosterone-fueled roots.",
                "criticsScore": "65%",
                "genres": "Sci-Fi, Action, Adventure, Mystery & Thriller",
                "metadata": "R, Released Jul 9 2010, 1h 47m",
                "visual": "https://resizing.flixster.com/3YsgAt6VfHoCqmYwsQzDz3S29SU=/206x305/v2/https://resizing.flixster.com/-XZAfHZM39UwaGJIFWKAE8fS0ak=/v3/t/assets/p8000437_p_v12_au.jpg"
            }
        }


class MovieResponse(BaseModel):
    """Response model for movie data with embedded vector values."""
    
    id: str
    values: List[float] = []
    metadata: Movie
    
    class Config:
        extra = "ignore"  # Ignore extra fields
        json_schema_extra = {
            "example": {
                "id": "movie-123",
                "values": [0.1, 0.2, 0.3, 0.4, 0.5],
                "metadata": {
                    "title": "Predators",
                    "description": "Brought together on a mysterious planet...",
                    "audienceConsensus": "N/A",
                    "audienceScore": "52%",
                    "audienceVerifiedCount": "100,000+ Ratings",
                    "cast": "Nimród Antal, Adrien Brody, Topher Grace...",
                    "criticReviews": "201 Reviews",
                    "criticsConsensus": "After a string of subpar sequels...",
                    "criticsScore": "65%",
                    "genres": "Sci-Fi, Action, Adventure, Mystery & Thriller",
                    "metadata": "R, Released Jul 9 2010, 1h 47m",
                    "visual": "https://resizing.flixster.com/3YsgAt6VfHoCqmYwsQzDz3S29SU=/206x305/v2/https://resizing.flixster.com/-XZAfHZM39UwaGJIFWKAE8fS0ak=/v3/t/assets/p8000437_p_v12_au.jpg"
                }
            }
        }