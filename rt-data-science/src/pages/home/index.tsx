import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getMovieList } from "../../services/homeService";
import MovieCard from "../../components/MovieCard";
import LoadingPage from "../../components/Loading";
import HeaderComponent from "../../components/Header";

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

const HomePage = () => {
  const [movies, setMovies] = useState<Movie[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(0);
  const [paginationTokens, setPaginationTokens] = useState<string[]>(() => {
    const savedTokens = localStorage.getItem("paginationTokens");
    try {
      const tokens = savedTokens ? JSON.parse(savedTokens) : [""];
      return Array.isArray(tokens)
        ? tokens.map((t) => (t == null ? "" : t))
        : [""];
    } catch (err) {
      console.error("Lỗi khi khôi phục paginationTokens:", err);
      return [""];
    }
  });
  const [searchQuery, setSearchQuery] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    localStorage.setItem("paginationTokens", JSON.stringify(paginationTokens));
  }, [paginationTokens]);

  useEffect(() => {
    const fetchMovies = async () => {
      setLoading(true);
      const cacheKey = `movies_page_${currentPage}`;
      const cached = localStorage.getItem(cacheKey);
      if (cached) {
        try {
          setMovies(JSON.parse(cached));
        } catch (err) {
          console.error(`Lỗi khi phân tích cache (${cacheKey}):`, err);
        }
        setLoading(false);
        return;
      }

      const token = paginationTokens[currentPage] || "";
      try {
        const data = await getMovieList(token);
        const moviesArray = Array.isArray(data.movies) ? data.movies : [];
        const mappedMovies = moviesArray.map((item: any) => {
          const metadataParts = item.metadata.metadata?.split(", ") || [];
          return {
            slug: item.id,
            title: item.metadata.title || "",
            posterSrc: item.metadata.visual || "",
            genres: item.metadata.genres || "",
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
        localStorage.setItem(cacheKey, JSON.stringify(mappedMovies));

        if (
          data.pagination_token &&
          !paginationTokens.includes(data.pagination_token)
        ) {
          setPaginationTokens((prev) => {
            const newTokens = [...prev];
            newTokens[currentPage + 1] = data.pagination_token;
            return newTokens;
          });
        }
      } catch (err) {
        console.error("Lỗi khi lấy phim:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchMovies();
  }, [currentPage, paginationTokens]);

  const getFilteredMovies = () => {
    if (!searchQuery) return movies;
    const allMovies: Movie[] = [];
    for (let i = 0; ; i++) {
      const cacheKey = `movies_page_${i}`;
      const cached = localStorage.getItem(cacheKey);
      if (!cached) break;
      try {
        const pageMovies = JSON.parse(cached);
        if (Array.isArray(pageMovies)) {
          allMovies.push(...pageMovies);
        }
      } catch (err) {
        console.error(
          `Lỗi khi phân tích dữ liệu từ localStorage (${cacheKey}):`,
          err
        );
      }
    }
    return allMovies.filter((movie) =>
      movie.title.toLowerCase().startsWith(searchQuery.toLowerCase())
    );
  };

  const filteredMovies = getFilteredMovies();

  const handleMovieClick = (movie: Movie) => {
    navigate(`/movie/${movie.slug}`, { state: { movie } });
  };

  const handleNextPage = () => {
    setCurrentPage((prev) => prev + 1);
  };

  const handlePrevPage = () => {
    setCurrentPage((prev) => prev - 1);
  };

  return (
    <div className="flex flex-col self-stretch min-h-dvh max-h-full bg-gray-100 gap-8">
      <HeaderComponent onSearch={setSearchQuery} />
      <div className="flex flex-col items-center px-4">
        <span className="text-red-600 text-3xl mb-8 font-bold">
          Movies in Theaters
        </span>
        {loading ? (
          <LoadingPage />
        ) : !filteredMovies.length ? (
          <div className="text-center py-10 text-gray-600">
            {searchQuery
              ? "Không tìm thấy phim khớp với tìm kiếm."
              : "Không có phim để hiển thị. Vui lòng kiểm tra API."}
          </div>
        ) : (
          <div className="w-full max-w-7xl">
            {[0, 1].map((row) => (
              <div
                key={row}
                className="grid grid-cols-5 gap-4 mb-14 justify-center"
              >
                {filteredMovies.slice(row * 5, row * 5 + 5).map((movie) => (
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
        {!searchQuery && (
          <div className="flex gap-4 mt-4">
            <button
              onClick={handlePrevPage}
              disabled={currentPage === 0}
              className="px-4 py-2 bg-gray-300 rounded"
            >
              Trước
            </button>
            <button
              onClick={handleNextPage}
              disabled={!paginationTokens[currentPage + 1]}
              className="px-4 py-2 bg-gray-300 rounded"
            >
              Sau
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default HomePage;
