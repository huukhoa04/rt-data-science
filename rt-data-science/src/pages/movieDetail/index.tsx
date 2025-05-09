import { useLocation } from "react-router-dom";
import { useNavigate } from "react-router-dom";

interface Movie {
  slug: string;
  title: string;
  posterSrc: string;
  genres: string;
  rated: string;
  runtime: string;
  release: string;
  cast: string;
  criticsScore: string;
  audienceScore: string;
  criticsReviews: string;
  audienceRatings: string;
  description: string;
  criticsConsensus: string;
}

const MovieDetailPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const movie = location.state?.movie as Movie;

  if (!movie) {
    return (
      <div className="text-center py-10 text-gray-600">
        Không tìm thấy thông tin phim.
      </div>
    );
  }

  return (
    <div className="flex flex-col bg-[#EFEFEF] gap-8">
      {/* Header */}
      <div className="flex justify-between items-center px-50 pt-14">
        <div className="text-2xl font-bold ">
          <button onClick={() => navigate("/home")}>{"<"} Rotten Tomato</button>
        </div>
        <div className="flex space-x-6">
          <button onClick={() => alert("Pressed!")}>Option</button>
          <button onClick={() => alert("Pressed!")}>Option</button>
          <button onClick={() => alert("Pressed!")}>Option</button>
        </div>
      </div>

      {/* Movie Details */}
      <div className="flex items-start self-stretch mx-48">
        {/* Left: Poster and Info */}
        <div className="flex-1">
          <div className="flex h-[500px] shrink-0">
            <img
              src={movie.posterSrc}
              alt={movie.title}
              className="mr-10 w-[450px] object-fill rounded-2xl"
            />
            <div className="flex flex-1 flex-col items-start bg-[#DAD8D8] pl-3 pt-3 gap-1 rounded-2xl">
              <span className="text-5xl font-bold">{movie.title}</span>
              <div className="flex items-center pr-[3px]">
                <span className="text-[16px] mr-[23px]">Genres:</span>
                <span className="text-[16px]">{movie.genres}</span>
              </div>
              <div className="flex items-start pr-[4px]">
                <span className="text-[16px] mb-[1px] mr-[28px]">Rated:</span>
                <span className="text-[16px]">{movie.rated}</span>
              </div>
              <div className="flex items-start pr-[4px]">
                <span className="text-[16px] mb-[1px] mr-[10px]">Runtime:</span>
                <span className="text-[16px]">{movie.runtime}</span>
              </div>
              <div className="flex items-start pr-[4px]">
                <span className="text-[16px] mb-[1px] mr-[14px]">Release:</span>
                <span className="text-[16px]">{movie.release}</span>
              </div>
              <div className="flex items-start pr-[3px]">
                <span className="text-[16px] mr-[44px]">Cast:</span>
                <span className="text-[16px] w-[255px]">{movie.cast}</span>
              </div>
              <div className="flex items-start self-stretch pl-[79px] pr-[79px] ml-[10px] mr-[10px]">
                <div className="flex flex-col shrink-0 items-center pt-[4px] pb-[4px] mr-[20px]">
                  <button
                    className="flex flex-col items-start bg-[#F63434] text-left pt-[12px] pb-[12px] pl-[11px] pr-[11px] mb-[4px] rounded-[4px] border-0"
                    onClick={() => alert("Pressed!")}
                  >
                    <span className="text-[20px]">{movie.criticsScore}</span>
                  </button>
                  <span className="text-[16px] font-bold mb-[4px]">
                    Critics Score
                  </span>
                  <span className="text-[#1A85EA] text-[10px]">
                    {movie.criticsReviews}
                  </span>
                </div>
                <div className="flex flex-1 flex-col pt-[4px] pb-[4px]">
                  <div className="flex flex-col self-stretch bg-[#5CEE27] pt-[13px] pb-[13px] mb-[4px] ml-[28px] mr-[28px] rounded-[2px]">
                    <span className="text-[20px] text-center ml-[12px] mr-[12px]">
                      {movie.audienceScore}
                    </span>
                  </div>
                  <span className="text-[16px] font-bold mb-[4px]">
                    Audience Score
                  </span>
                  <span className="text-[#1A85EA] text-[10px] ml-[19px] mr-[19px]">
                    {movie.audienceRatings}
                  </span>
                </div>
              </div>
            </div>
          </div>
          <div className="flex flex-col items-start self-stretch pt-[10px] pb-[10px]">
            <span className="text-[36px] mb-[11px]">Description</span>
            <span className="text-[20px]">{movie.description}</span>
          </div>
        </div>

        {/* Right: Critics Consensus */}
        <div className="flex flex-col shrink-0 items-start pt-[8px] pb-[8px]">
          <div className="flex items-start pr-[3px] mb-[10px] ml-[13px]">
            {movie.criticsConsensus && (
              <img
                src={movie.criticsConsensus}
                alt="Critics Consensus Icon"
                className="w-[34px] h-[32px] mr-[10px] object-fill"
              />
            )}
            <span className="text-[32px] font-bold">Critics Consensus</span>
          </div>
          <span className="text-[20px] w-[298px] mb-[10px] ml-[13px]">
            {movie.criticsConsensus}
          </span>
          <span className="text-[#1A85EA] text-[16px] ml-[13px]">
            Read Critics Reviews
          </span>
        </div>
      </div>
    </div>
  );
};

export default MovieDetailPage;
