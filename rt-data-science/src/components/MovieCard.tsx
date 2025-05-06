// src/components/MovieCard.jsx
import { Link } from 'react-router-dom';

// Định nghĩa kiểu cho props
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
      className="flex flex-1 flex-col items-start pt-[10px] pb-[10px] mr-[54px] rounded-[8px] border-[1px] border-solid border-[#202023]"
    >
      <img
        src={imageSrc}
        alt={title}
        className="self-stretch h-[223px] mb-[8px] ml-[10px] mr-[10px] rounded-[8px] object-fill"
      />
      <span className="text-[#000000] text-[18px] font-bold mb-[8px] ml-[10px]">
        {title}
      </span>
      <span className="text-[#000000] text-[10px] ml-[10px]">{genres}</span>
    </Link>
  );
};

export default MovieCard;