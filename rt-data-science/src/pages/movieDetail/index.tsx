import { useParams, useNavigate } from "react-router-dom";

// Định nghĩa kiểu cho dữ liệu phim
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
  consensusIconSrc: string;
}

const MovieDetailPage = () => {
  const { slug } = useParams<{ slug: string }>(); // Lấy slug từ URL
  const navigate = useNavigate();

  // Dữ liệu mẫu (thay bằng API nếu cần)
  const movie: Movie = {
    slug: "thunderbolts-1",
    title: "ThunderBolts",
    posterSrc:
      "https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/9a99d7b5-a38f-4fb8-b225-dbea2f3c46fd",
    genres: "Action, Adventure, Crime, Drama",
    rated: "PG-13",
    runtime: "2h 6m",
    release: "May 1, 2025",
    cast: "Florence Pugh, Sebastian Stan, David Harbour, Wyatt Russell, Olga Kurylenko, Lewis Pullman, Geraldine Viswanathan, Chris Bauer",
    criticsScore: "82%",
    audienceScore: "90%",
    criticsReviews: "150 reviews",
    audienceRatings: "100.000+ Ratings",
    description:
      'In "Thunderbolts*," Marvel Studios assembles an unconventional team of antiheroes -- Yelena Belova, Bucky Barnes, Red Guardian, Ghost, Taskmaster and John Walker. After finding themselves ensnared in a death trap set by Valentina Allegra de Fontaine, these disillusioned castoffs must embark on a dangerous mission that will force them to confront the darkest corners of their pasts. Will this dysfunctional group tear themselves apart, or find redemption and unite as something much more before it\'s too late?',
    criticsConsensus:
      "Assembling a ragtag band of underdogs with Florence Pugh as their magnetic standout, Thunderbolts* refreshingly returns to the tried-and-true blueprint of the MCU's best adventures.",
    consensusIconSrc:
      "https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/8ebb5ad1-b608-40df-b81d-5aed4784b640",
  };

  // Giả lập logic: nếu slug không khớp, trả về thông báo lỗi (thay bằng API 404)
  if (slug !== movie.slug) {
    return <div>Phim không tìm thấy!</div>;
  }

  return (
    <div className="flex flex-col bg-[# megrFFFFFF]">
      <div className="self-stretch bg-[#EFEFEF] h-[1027px]">
        {/* Header */}
        <div className="flex flex-col items-start self-stretch pt-[27px] pb-[12px] mb-[26px]">
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
              <span className="text-[#0B0B0B] text-[16px]">Option</span>
            </div>
          </div>
          <div className="flex justify-between items-start self-stretch pl-[274px] pr-[274px] ml-[144px] mr-[144px]">
            <button
              className="flex flex-col shrink-0 items-start bg-[#FFFFFF] text-left pt-[8px] pb-[8px] pl-[49px] pr-[49px] rounded-[6px] border-[1px] border-solid border-[#000000] shadow-[0px_4px_4px_#00000040]"
              onClick={() => navigate("/recommend")}
            >
              <span className="text-[#000000] text-[16px] font-bold">
                Gợi ý phim
              </span>
            </button>
            <button
              className="flex flex-col shrink-0 items-start bg-[#FFFFFF] text-left pt-[8px] pb-[8px] pl-[57px] pr-[57px] rounded-[6px] border-[1px] border-solid border-[#000000] shadow-[0px_4px_4px_#00000040]"
              onClick={() => alert("Pressed!")}
            >
              <span className="text-[#000000] text-[16px] font-bold">
                Phổ biến
              </span>
            </button>
            <button
              className="flex flex-col shrink-0 items-start bg-[#FFFFFF] text-left pt-[8px] pb-[8px] pl-[48px] pr-[48px] rounded-[6px] border-[1px] border-solid border-[#000000] shadow-[0px_4px_4px_#00000040]"
              onClick={() => alert("Pressed!")}
            >
              <span className="text-[#000000] text-[16px] font-bold">
                Mới ra mắt
              </span>
            </button>
          </div>
        </div>

        {/* Movie Details */}
        <div className="flex items-start self-stretch pt-[16px] pb-[191px] pl-[64px] pr-[64px] mb-[14px] ml-[80px] mr-[80px]">
          {/* Left: Poster and Info */}
          <div className="flex-1 pt-[8px] pb-[8px] mr-[12px]">
            <div className="flex items-start self-stretch mb-[24px]">
              <img
                src={movie.posterSrc}
                alt={movie.title}
                className="w-[316px] h-[392px] mr-[24px] object-fill"
              />
              <div className="flex flex-1 flex-col items-start bg-[#DAD8D8] pt-[10px] pb-[27px] rounded-[4px]">
                <span className="text-[#000000] text-[60px] font-bold mb-[12px] ml-[10px] mr-[10px]">
                  {movie.title}
                </span>
                <div className="flex items-center pr-[3px] mb-[12px] ml-[10px]">
                  <span className="text-[#000000] text-[18px] mr-[23px]">
                    Genres:
                  </span>
                  <span className="text-[#000000] text-[18px]">
                    {movie.genres}
                  </span>
                </div>
                <div className="flex items-start pr-[4px] mb-[12px] ml-[10px]">
                  <span className="text-[#000000] text-[14px] mb-[1px] mr-[28px]">
                    Rated:
                  </span>
                  <span className="text-[#000000] text-[18px]">
                    {movie.rated}
                  </span>
                </div>
                <div className="flex items-start pr-[4px] mb+[12px] ml-[10px]">
                  <span className="text-[#000000] text-[14px] mb-[1px] mr-[10px]">
                    Runtime:
                  </span>
                  <span className="text-[#000000] text-[18px]">
                    {movie.runtime}
                  </span>
                </div>
                <div className="flex items-start pr-[4px] mb-[12px] ml-[10px]">
                  <span className="text-[#000000] text-[14px] mb-[1px] mr-[14px]">
                    Release:
                  </span>
                  <span className="text-[#000000] text-[18px]">
                    {movie.release}
                  </span>
                </div>
                <div className="flex items-start pr-[3px] mb-[12px] ml-[10px]">
                  <span className="text-[#000000] text-[16px] mr-[44px]">
                    Cast:
                  </span>
                  <span className="text-[#000000] text-[16px] w-[255px]">
                    {movie.cast}
                  </span>
                </div>
                <div className="flex items-start self-stretch pl-[79px] pr-[79px] ml-[10px] mr-[10px]">
                  <div className="flex flex-col shrink-0 items-center pt-[4px] pb-[4px] mr-[20px]">
                    <button
                      className="flex flex-col items-start bg-[#F63434] text-left pt-[12px] pb-[12px] pl-[11px] pr-[11px] mb-[4px] rounded-[4px] border-0"
                      onClick={() => alert("Pressed!")}
                    >
                      <span className="text-[#000000] text-[20px]">
                        {movie.criticsScore}
                      </span>
                    </button>
                    <span className="text-[#000000] text-[16px] font-bold mb-[4px]">
                      Critics Score
                    </span>
                    <span className="text-[#1A85EA] text-[10px]">
                      {movie.criticsReviews}
                    </span>
                  </div>
                  <div className="flex flex-1 flex-col pt-[4px] pb-[4px]">
                    <div className="flex flex-col self-stretch bg-[#5CEE27] pt-[13px] pb-[13px] mb-[4px] ml-[28px] mr-[28px] rounded-[2px]">
                      <span className="text-[#000000] text-[20px] text-center ml-[12px] mr-[12px]">
                        {movie.audienceScore}
                      </span>
                    </div>
                    <span className="text-[#000000] text-[16px] font-bold mb-[4px]">
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
              <span className="text-[#000000] text-[36px] mb-[11px]">
                Description
              </span>
              <span className="text-[#000000] text-[20px]">
                {movie.description}
              </span>
            </div>
          </div>

          {/* Right: Critics Consensus */}
          <div className="flex flex-col shrink-0 items-start pt-[8px] pb-[8px]">
            <div className="flex items-start pr-[3px] mb-[10px] ml-[13px]">
              <img
                src={movie.consensusIconSrc}
                alt="Critics Consensus Icon"
                className="w-[34px] h-[32px] mr-[10px] object-fill"
              />
              <span className="text-[#000000] text-[32px] font-bold">
                Critics Consensus
              </span>
            </div>
            <span className="text-[#000000] text-[20px] w-[298px] mb-[10px] ml-[13px]">
              {movie.criticsConsensus}
            </span>
            <span className="text-[#1A85EA] text-[16px] ml-[13px]">
              Read Critics Reviews
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MovieDetailPage;
