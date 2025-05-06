import MovieCard from '../../components/MovieCard'

// Định nghĩa kiểu cho phim (dùng cho MovieCard)
interface Movie {
  slug: string;
  imageSrc: string;
  title: string;
  genres: string;
}

// Định nghĩa kiểu cho ảnh đơn (phần thứ hai trong Kết quả tìm kiếm)
interface ImageCard {
  imageSrc: string;
}

const RecommendPage = () => {
  // Dữ liệu mẫu cho phim (thay bằng API)
  const movies: Movie[] = [
    {
      slug: 'thunderbolts-1',
      imageSrc:
        'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/21739557-ae88-470d-a23f-e91d18daae5a',
      title: 'ThunderBolts',
      genres: 'Action, Adventure, Crime, Drama',
    },
    {
      slug: 'thunderbolts-2',
      imageSrc:
        'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/e2331a82-6a4e-4a82-901a-bd57627e38b9',
      title: 'ThunderBolts',
      genres: 'Action, Adventure, Crime, Drama',
    },
    {
      slug: 'thunderbolts-3',
      imageSrc:
        'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/7461a8b3-ff1a-4be0-adea-d6ddb45a581b',
      title: 'ThunderBolts',
      genres: 'Action, Adventure, Crime, Drama',
    },
    {
      slug: 'thunderbolts-4',
      imageSrc:
        'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/b524e976-6291-4a3e-84dc-43e9b6d8407f',
      title: 'ThunderBolts',
      genres: 'Action, Adventure, Crime, Drama',
    },
    {
      slug: 'thunderbolts-5',
      imageSrc:
        'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/35380430-e308-4e4b-afa1-deda61309ac2',
      title: 'ThunderBolts',
      genres: 'Action, Adventure, Crime, Drama',
    },
  ];

  // Dữ liệu mẫu cho phần ảnh đơn
  const imageCards: ImageCard[] = [
    {
      imageSrc:
        'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/6fe0609f-1d3e-49b2-a3c9-e7d921665906',
    },
    {
      imageSrc:
        'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/8d51523a-b256-4b97-8a9f-695ab942dc36',
    },
    {
      imageSrc:
        'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/2285d191-2e33-4d07-8ef9-4d775643cdca',
    },
    {
      imageSrc:
        'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/8ea5ee6d-c91b-488c-9904-97176423ed4d',
    },
    {
      imageSrc:
        'https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/1669aca1-b68f-4cb2-9e73-15e4ebbd2b68',
    },
  ];

  return (
    <div className="flex flex-col bg-[#FFFFFF]">
      <div className="flex flex-col items-start self-stretch bg-[#EFEFEF] h-[1024px]">
        {/* Header */}
        <div className="flex items-start mt-[27px] mb-[55px] ml-[144px]">
          <span className="text-[#0B0B0B] text-[20px] mr-[845px]">
            Rotten Tomatoes
          </span>
          <div className="flex shrink-0 items-center pr-[3px]">
            <span className="text-[#0B0B0B] text-[16px] mr-[23px]">
              Option
            </span>
            <span className="text-[#0B0B0B] text-[16px] mr-[23px]">
              Option
            </span>
            <span className="text-[#0B0B0B] text-[16px]">Option</span>
          </div>
        </div>

        {/* Input Section */}
        <div className="flex flex-col items-center self-stretch pt-[10px] pb-[10px] pl-[8px] pr-[8px] mb-[14px] ml-[144px] mr-[114px]">
          <span className="text-[#000000] text-[30px] font-bold mb-[10px]">
            Nhập yêu cầu của bạn:
          </span>
          <div className="flex flex-col items-start self-stretch bg-[#FFFFFF] pt-[8px] pb-[110px] mb-[10px] rounded-[6px] border-[1px] border-solid border-[#CBD5E1]">
            <span className="text-[#94A3B8] text-[20px] ml-[12px]">
              Ví dụ: Tôi muốn xem phim hành động có điểm IMDb trên 8
            </span>
          </div>
          <button
            className="flex flex-col items-start bg-[#0F172A] text-left pt-[12px] pb-[12px] pl-[21px] pr-[21px] rounded-[6px] border-0"
            onClick={() => alert('Pressed!')}
          >
            <span className="text-[#FFFFFF] text-[20px] font-bold">
              Send message
            </span>
          </button>
        </div>

        {/* Suggestion Buttons */}
        <div className="flex items-start pt-[24px] pb-[24px] pl-[8px] pr-[8px] mb-[27px] ml-[144px]">
          <button
            className="flex flex-col shrink-0 items-start bg-[#FFFFFF] text-left pt-[7px] pb-[7px] pl-[13px] pr-[13px] mr-[10px] rounded-[6px] border-0 shadow-[0px_2px_4px_#1E293B40]"
            onClick={() => alert('Pressed!')}
          >
            <span className="text-[#000000] text-[14px] font-bold">
              Phim hành động rating cao
            </span>
          </button>
          <button
            className="flex flex-col shrink-0 items-start bg-[#FFFFFF] text-left pt-[7px] pb-[7px] pl-[13px] pr-[13px] mr-[10px] rounded-[6px] border-0 shadow-[0px_2px_4px_#1E293B40]"
            onClick={() => alert('Pressed!')}
          >
            <span className="text-[#000000] text-[14px] font-bold">
              Phim do Christopher Nolan đạo diễn
            </span>
          </button>
          <button
            className="flex flex-col shrink-0 items-start bg-[#FFFFFF] text-left pt-[7px] pb-[7px] pl-[13px] pr-[13px] mr-[10px] rounded-[6px] border-0 shadow-[0px_2px_4px_#1E293B40]"
            onClick={() => alert('Pressed!')}
          >
            <span className="text-[#000000] text-[14px] font-bold">
              Phim của Marvel Studio
            </span>
          </button>
          <button
            className="flex flex-col shrink-0 items-start bg-[#FFFFFF] text-left pt-[7px] pb-[7px] pl-[13px] pr-[13px] mr-[10px] rounded-[6px] border-0 shadow-[0px_2px_4px_#1E293B40]"
            onClick={() => alert('Pressed!')}
          >
            <span className="text-[#000000] text-[14px] font-bold">
              Phim năm 2020 trở lên
            </span>
          </button>
          <button
            className="flex flex-col shrink-0 items-start bg-[#FFFFFF] text-left pt-[7px] pb-[7px] pl-[13px] pr-[13px] rounded-[6px] border-0 shadow-[0px_2px_4px_#1E293B40]"
            onClick={() => alert('Pressed!')}
          >
            <span className="text-[#000000] text-[14px] font-bold">
              Phim hoạt hình hay cho cả gia đình
            </span>
          </button>
        </div>

        {/* Search Results */}
        <span className="text-[#AC3434] text-[32px] mb-[27px] ml-[144px]">
          Kết quả tìm kiếm:
        </span>
        <div className="self-stretch ml-[144px] mr-[114px]">
          {/* Movie Cards */}
          <div className="flex items-start self-stretch mb-[54px]">
            {movies.map((movie, index) => (
              <MovieCard
                key={index}
                imageSrc={movie.imageSrc}
                title={movie.title}
                genres={movie.genres}
                slug={movie.slug}
              />
            ))}
          </div>
          {/* Image-only Cards */}
          <div className="flex items-start self-stretch">
            {imageCards.map((image, index) => (
              <div
                key={index}
                className="flex flex-1 flex-col pt-[10px] mr-[54px] rounded-[8px] border-[1px] border-solid border-[#202023]"
              >
                <img
                  src={image.imageSrc}
                  alt={`Image ${index + 1}`}
                  className="self-stretch h-[97px] ml-[10px] mr-[10px] rounded-[8px] object-fill"
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecommendPage;