// src/pages/home.jsx
import { useNavigate } from 'react-router-dom';
import MovieCard from '../../components/MovieCard';

const HomePage = () => {
  const navigate = useNavigate();

  // Sample movie data (replace with actual data from API or state)
  const movies = [
    {
      slug: 'thunderbolts-1',
      imageSrc: 'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/f2287e79-9116-4173-90c9-3109ddc9dfb5',
      title: 'ThunderBolts',
      genres: 'Action, Adventure, Crime, Drama',
    },
    {
      slug: 'thunderbolts-2',
      imageSrc: 'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/1b31e70e-c48e-43a8-a1f2-13e8872e1c25',
      title: 'ThunderBolts',
      genres: 'Action, Adventure, Crime, Drama',
    },
    {
      slug: 'thunderbolts-3',
      imageSrc: 'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/6d976a51-a0d9-4d10-863b-65b63bcbd67f',
      title: 'ThunderBolts',
      genres: 'Action, Adventure, Crime, Drama',
    },
    {
      slug: 'thunderbolts-4',
      imageSrc: 'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/bfedbc7f-3e9f-40cd-ae62-2e5d6e3fdbae',
      title: 'ThunderBolts',
      genres: 'Action, Adventure, Crime, Drama',
    },
    {
      slug: 'thunderbolts-5',
      imageSrc: 'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/faa92b4d-0787-447a-aa6a-8300a758e5a6',
      title: 'ThunderBolts',
      genres: 'Action, Adventure, Crime, Drama',
    },
    {
      slug: 'thunderbolts-6',
      imageSrc: 'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/f1926856-b0ef-41ab-8f94-08adebd1a479',
      title: 'ThunderBolts',
      genres: 'Action, Adventure, Crime, Drama',
    },
    {
      slug: 'thunderbolts-7',
      imageSrc: 'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/def964bb-fe98-4ce9-be7e-5947327bd43e',
      title: 'ThunderBolts',
      genres: 'Action, Adventure, Crime, Drama',
    },
    {
      slug: 'thunderbolts-8',
      imageSrc: 'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/455e1408-0b72-4f75-9e64-f69410757bef',
      title: 'ThunderBolts',
      genres: 'Action, Adventure, Crime, Drama',
    },
    {
      slug: 'thunderbolts-9',
      imageSrc: 'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/24e07565-2000-47bd-86ac-984227cac0c8',
      title: 'ThunderBolts',
      genres: 'Action, Adventure, Crime, Drama',
    },
    {
      slug: 'thunderbolts-10',
      imageSrc: 'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/5d96ef78-9380-435a-8f26-b0fd71e8b7c3',
      title: 'ThunderBolts',
      genres: 'Action, Adventure, Crime, Drama',
    },
  ];

  return (
    <div className="flex flex-col bg-[#FFFFFF]">
      <div className="self-stretch bg-[#EFEFEF] h-[1030px]">
        <div className="flex flex-col items-start self-stretch pt-[27px] pb-[12px] mb-[32px]">
          <div className="flex items-start mb-[20px] ml-[144px]">
            <span className="text-[#0B0B0B] text-[20px] mr-[183px]">
              Rotten Tomatoes
            </span>
            <input
              type="text"
              placeholder="Search"
              className="shrink-0 text-[#1E1E1E] bg-[#FFFFFF] text-[16px] pt-[12px] pb-[12px] pl-[16px] pr-[16px] mr-[157px] rounded-[9999px] border-[1px] border-solid border-[#D9D9D9]"
            />
            <div className="flex shrink-0 items-center pr-[3px]">
              <span className="text-[#0B0B0B] text-[16px] mr-[23px]">
                Option
              </span>
              <span className="text-[#0B0B0B] text-[16px] mr-[23px]">
                Option
              </span>
              <span className="text-[#0B0B0B] text-[16px]">
                Option
              </span>
            </div>
          </div>
          <div className="flex justify-between items-start self-stretch pl-[274px] pr-[274px] ml-[144px] mr-[144px]">
            <button
              className="flex flex-col shrink-0 items-start bg-[#FFFFFF] text-left pt-[8px] pb-[8px] pl-[49px] pr-[49px] rounded-[6px] border-[1px] border-solid border-[#000000] shadow-[0px_4px_4px_#00000040]"
              onClick={() => navigate('/recommend')}
            >
              <span className="text-[#000000] text-[16px] font-bold">
                Gợi ý phim
              </span>
            </button>
            <button
              className="flex flex-col shrink-0 items-start bg-[#FFFFFF] text-left pt-[8px] pb-[8px] pl-[57px] pr-[57px] rounded-[6px] border-[1px] border-solid border-[#000000] shadow-[0px_4px_4px_#00000040]"
              onClick={() => alert('Pressed!')}
            >
              <span className="text-[#000000] text-[16px] font-bold">
                Phổ biến
              </span>
            </button>
            <button
              className="flex flex-col shrink-0 items-start bg-[#FFFFFF] text-left pt-[8px] pb-[8px] pl-[48px] pr-[48px] rounded-[6px] border-[1px] border-solid border-[#000000] shadow-[0px_4px_4px_#00000040]"
              onClick={() => alert('Pressed!')}
            >
              <span className="text-[#000000] text-[16px] font-bold">
                Mới ra mắt
              </span>
            </button>
          </div>
        </div>
        <div className="flex flex-col items-start self-stretch pt-[16px] pb-[16px] mb-[131px] ml-[81px] mr-[49px]">
          <span className="text-[#AC3434] text-[32px] mb-[33px] ml-[64px]">
            Movies in Theaters
          </span>
          <div className="self-stretch ml-[64px] mr-[64px]">
            <div className="flex items-start self-stretch mb-[54px]">
              {movies.slice(0, 5).map((movie, index) => (
                <MovieCard
                  key={index}
                  imageSrc={movie.imageSrc}
                  title={movie.title}
                  genres={movie.genres}
                  slug={movie.slug}
                />
              ))}
            </div>
            <div className="flex items-start self-stretch">
              {movies.slice(5).map((movie, index) => (
                <MovieCard
                  key={index + 5}
                  imageSrc={movie.imageSrc}
                  title={movie.title}
                  genres={movie.genres}
                  slug={movie.slug}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;