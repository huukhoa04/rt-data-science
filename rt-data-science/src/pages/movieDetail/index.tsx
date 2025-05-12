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
    <div className="flex flex-col bg-[#EFEFEF] h-full w-full gap-8">
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
      <div className="flex items-start self-stretch mx-40">
        {/* Left*/}
        <div className="flex flex-col items-start ">
          {/* Movie Poster and Info */}
          <div className="flex h-fit gap-8">
            <img
              src={movie.posterSrc}
              alt={movie.title}
              className="w-[250px] h-fit rounded-2xl"
            />
            <div className="flex flex-1 h-fit flex-col items-start bg-[#DAD8D8] pl-3 pr-5 pt-3 gap-2 rounded-2xl">
              <span className="text-4xl w-fit h-fit font-bold">
                {movie.title}
              </span>
              <div className="flex items-start w-fit h-fit">
                <span className="mr-[13px]">Genres:</span>
                <span>{movie.genres}</span>
              </div>
              <div className="flex items-start w-fit h-fit">
                <span className="mr-[22px]">Rated:</span>
                <span>{movie.rated}</span>
              </div>
              <div className="flex items-start w-fit h-fit">
                <span className="mr-[5px]">Runtime:</span>
                <span>{movie.runtime}</span>
              </div>
              <div className="flex items-start w-fit h-fit">
                <span className="mr-[10px]">Release:</span>
                <span>{movie.release}</span>
              </div>
              <div className="flex items-start w-full h-fit">
                <span className="mr-[32px]">Cast:</span>
                <span className="text-justify ">{movie.cast}</span>
              </div>

              {/* Ratings */}
              <div className="flex h-fit w-full justify-center my-3 gap-8">
                <div className="flex flex-col items-center gap-2">
                  <div className="flex items-center bg-[#F63434] px-2 py-3 rounded-[4px]">
                    <span className="text-[20px]">{movie.criticsScore}</span>
                  </div>
                  <span className="font-bold">Critics Score</span>
                  <span className="text-[#1A85EA] text-[10px]">
                    {movie.criticsReviews}
                  </span>
                </div>
                <div className="flex flex-col items-center gap-2">
                  <div className="flex items-center bg-[#5CEE27] px-2 py-3 rounded-[4px]]">
                    <span className="text-[20px]">{movie.audienceScore}</span>
                  </div>
                  <span className="font-bold">Audience Score</span>
                  <span className="text-[#1A85EA] text-[10px]">
                    {movie.audienceRatings}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Description */}
          <div className="flex flex-col items-start mb-10 w-full h-fit gap-3">
            <span className="text-3xl font-bold">Description</span>
            <span className="text-justify">{movie.description}</span>
          </div>
        </div>

        {/* Right: Critics Consensus */}
        <div className="flex flex-col items-start w-[1750px] h-fit ml-8 gap-3">
          <div className="flex items-center gap-2">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="green"
              className="size-6 inline "
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9 12.75 11.25 15 15 9.75M21 12c0 1.268-.63 2.39-1.593 3.068a3.745 3.745 0 0 1-1.043 3.296 3.745 3.745 0 0 1-3.296 1.043A3.745 3.745 0 0 1 12 21c-1.268 0-2.39-.63-3.068-1.593a3.746 3.746 0 0 1-3.296-1.043 3.745 3.745 0 0 1-1.043-3.296A3.745 3.745 0 0 1 3 12c0-1.268.63-2.39 1.593-3.068a3.745 3.745 0 0 1 1.043-3.296 3.746 3.746 0 0 1 3.296-1.043A3.746 3.746 0 0 1 12 3c1.268 0 2.39.63 3.068 1.593a3.746 3.746 0 0 1 3.296 1.043 3.746 3.746 0 0 1 1.043 3.296A3.745 3.745 0 0 1 21 12Z"
              />
            </svg>
            <span className="text-4xl font-bold">Critics Consensus</span>
          </div>
          <span className="w-fit text-justify">{movie.criticsConsensus}</span>
          <a href="" className="text-[#1A85EA] w-full text-center">Read Critics Reviews</a>
        </div>
      </div>
    </div>
  );
};

export default MovieDetailPage;
