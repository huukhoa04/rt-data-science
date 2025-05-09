import { Link } from "react-router-dom";

interface MovieCardProps {
  imageSrc: string;
  title: string;
  genres: string;
  slug: string;
}

const MovieCard = ({ imageSrc, title, genres, slug }: MovieCardProps) => {
  return (
    <Link
      to={`/movie/${slug}`}
      className="flex flex-col bg-white rounded-xl shadow-md overflow-hidden border border-gray-300 hover:shadow-lg transition-shadow duration-200 box-border"
    >
      <div className="aspect-[2/3] overflow-hidden rounded-lg">
        <img src={imageSrc} alt={title} className="h-full object-cover" />
      </div>

      <div className="p-4">
        <h3 className="text-lg font-semibold text-black mb-2 line-clamp-2 overflow-hidden leading-snug min-h-[3rem]">
          {title}
        </h3>
        <p className="text-sm text-gray-600 line-clamp-1 overflow-hidden leading-snug min-h-[1.25rem]">
          {genres}
        </p>
      </div>
    </Link>
  );
};

export default MovieCard;
