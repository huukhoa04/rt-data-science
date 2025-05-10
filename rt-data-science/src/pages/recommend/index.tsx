import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import MovieCard from "../../components/MovieCard";
import LoadingPage from "../../components/Loading";

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

const RecommendPage = () => {
  const [movies, setMovies] = useState<Movie[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const navigate = useNavigate();

  const fetchMovies = async (searchQuery: string) => {
    setLoading(true);
    setError(null);
    try {
      const url = "http://localhost:5000/api/movies/recommend";
      const payload = {
        query: searchQuery || "recommended movies",
        top_k: 5,
      };
      const res = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        throw new Error(`Lỗi ${res.status}: ${res.statusText}`);
      }
      const data = await res.json();
      if (data.detail) {
        throw new Error(data.detail);
      }
      const moviesArray = Array.isArray(data)
        ? data
        : Array.isArray(data.movies)
        ? data.movies
        : [];
      const mappedMovies = moviesArray.map((item: any) => {
        const metadataParts = item.metadata.metadata.split(", ");
        return {
          slug: item.id,
          title: item.metadata.title,
          posterSrc: item.metadata.visual,
          genres: item.metadata.genres,
          rated: metadataParts[0] || "",
          runtime: metadataParts[2] || "",
          release: metadataParts[1] || "",
          cast: item.metadata.cast || "",
          criticsScore: item.metadata.criticsScore || "",
          audienceScore: item.metadata.audienceScore || "",
          criticsReviews: item.metadata.criticReviews || "",
          audienceRatings: item.metadata.audienceVerifiedCount || "",
          description: item.metadata.description || "",
          criticsConsensus: item.metadata.criticsConsensus || "",
        };
      });
      setMovies(mappedMovies);
    } catch (err) {
      setError(`Không thể tải phim gợi ý: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMovies("recommended movies");
  }, []);

  const handleMovieClick = (movie: Movie) => {
    navigate(`/movie/${movie.slug}`, { state: { movie } });
  };

  const handleSendMessage = () => {
    fetchMovies(query);
  };

  return (
    <div className="flex flex-col bg-gray-100 gap-8">
      {/* Header */}
      <div className="flex justify-between items-center px-50 pt-12">
        <div className="text-2xl font-bold whitespace-nowrap">
          <button onClick={() => navigate("/home")}>{"<"} Rotten Tomato</button>
        </div>
        <div className="flex space-x-6">
          <button onClick={() => alert("Pressed!")}>Option</button>
          <button onClick={() => alert("Pressed!")}>Option</button>
          <button onClick={() => alert("Pressed!")}>Option</button>
        </div>
      </div>

      {/* Input Section */}
      <div className="flex flex-col items-center gap-[8px] mx-48">
        <span className="text-3xl font-bold">Nhập yêu cầu của bạn:</span>
        <div className="flex items-start self-stretch bg-white h-70 rounded-[6px] border-1 border-solid border-[#CBD5E1]">
          <input
            className="text-gray w-full ml-4 mt-3 border-none outline-none"
            placeholder="Ví dụ: Tôi muốn xem phim hành động có điểm IMDb trên 8"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <button
          className="bg-[#0F172A] px-4 py-3 rounded-[6px]"
          onClick={handleSendMessage}
        >
          <span className="text-white text-2xl font-bold">Send message</span>
        </button>
      </div>

      {/* Suggestion Buttons */}
      <div className="flex items-center justify-between gap-3 mx-48">
        <button
          className="bg-white px-2 py-1 rounded-[6px] shadow-[0px_2px_4px_#1E293B40]"
          onClick={() => fetchMovies("Phim hành động rating cao")}
        >
          <span className="text-[14px] font-bold">
            Phim hành động rating cao
          </span>
        </button>
        <button
          className="bg-white px-2 py-1 rounded-[6px] shadow-[0px_2px_4px_#1E293B40]"
          onClick={() => fetchMovies("Phim do Christopher Nolan đạo diễn")}
        >
          <span className="text-[14px] font-bold">
            Phim do Christopher Nolan đạo diễn
          </span>
        </button>
        <button
          className="bg-white px-2 py-1 rounded-[6px] shadow-[0px_2px_4px_#1E293B40]"
          onClick={() => fetchMovies("Phim của Marvel Studio")}
        >
          <span className="text-[14px] font-bold">
            Phim của Marvel Studio
          </span>
        </button>
        <button
          className="bg-white px-2 py-1 rounded-[6px] shadow-[0px_2px_4px_#1E293B40]"
          onClick={() => fetchMovies("Phim năm 2020 trở lên")}
        >
          <span className="text-[14px] font-bold">
            Phim năm 2020 trở lên
          </span>
        </button>
        <button
          className="bg-white px-2 py-1 rounded-[6px] shadow-[0px_2px_4px_#1E293B40]"
          onClick={() => fetchMovies("Phim hoạt hình hay cho cả gia đình")}
        >
          <span className="text-[14px] font-bold">
            Phim hoạt hình hay cho cả gia đình
          </span>
        </button>
      </div>

      {/* Search Results */}
      <span className="text-red-600 text-3xl ml-48 font-bold">Results:</span>
      {loading ? (
        <LoadingPage />
      ) : error ? (
        <div className="text-center py-10 text-gray-600">{error}</div>
      ) : !movies.length ? (
        <div className="text-center py-10 text-gray-600">
          Không có phim gợi ý nào.
        </div>
      ) : (
        <div className="self-center w-[85.71%]">
          {[0, 1].map((row) => (
            <div
              key={row}
              className="grid grid-cols-5 gap-4 mb-14 justify-center"
            >
              {movies.slice(row * 5, row * 5 + 5).map((movie) => (
                <div
                  key={movie.slug}
                  onClick={() => handleMovieClick(movie)}
                  className="cursor-pointer"
                >
                  <MovieCard
                    imageSrc={movie.posterSrc}
                    title={movie.title}
                    genres={movie.genres}
                    slug={movie.slug}
                  />
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default RecommendPage;
