interface MovieCardProps {
  imageSrc: string;
  title: string;
  genres: string;
  slug: string;
  onClick?: () => void;
}

const MovieCard = ({ imageSrc, title, genres, onClick }: MovieCardProps) => {
  return (
    <div
      onClick={onClick}
      className="cursor-pointer flex flex-col bg-white rounded-xl shadow-md overflow-hidden border border-gray-300 hover:shadow-lg box-border"
    >
      <div className="aspect-[2/3] overflow-hidden rounded-lg">
        <img src={imageSrc} alt={title} className="h-full object-cover" />
      </div>

      <div className="p-4">
        <h3 className="text-lg font-semibold line-clamp-1 overflow-hidden min-h-[1.25rem] mb-2">
          {title}
        </h3>
        <p className="text-sm text-gray-600 line-clamp-1 overflow-hidden min-h-[1.25rem]">
          {genres}
        </p>
      </div>
    </div>
  );
};

export default MovieCard;
