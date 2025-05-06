import { useNavigate } from 'react-router-dom'; // Import useNavigate từ react-router-dom

export default function LandingPage() {
  const navigate = useNavigate(); // Khởi tạo navigate

  // Hàm handleClick để điều hướng đến trang home
  const handleClick = () => {
    navigate('/home'); // Điều hướng đến route /home
  };

  return (
    <>
      <div className="flex flex-col bg-[#FFFFFF]">
        <div className="flex flex-col self-stretch bg-[#0B0B0B] h-[1181px]">
          <div className="flex items-center self-stretch pt-[50px] pb-[50px] pl-[144px] pr-[144px]">
            <span className="flex-1 text-[#FFFFFF] text-[20px]">
              Rotten Tomatoes
            </span>
            <div className="flex shrink-0 items-center pr-[3px]">
              <span className="text-[#FFFFFF] text-[16px] mr-[23px]">
                Option
              </span>
              <span className="text-[#FFFFFF] text-[16px] mr-[23px]">
                Option
              </span>
              <span className="text-[#FFFFFF] text-[16px]">
                Option
              </span>
            </div>
          </div>
          <div className="flex items-center self-stretch pt-[53px] pb-[53px] pl-[64px] pr-[64px] mb-[323px] ml-[80px] mr-[80px]">
            <div className="flex flex-1 flex-col items-start mr-[12px]">
              <div className="flex flex-col items-start self-stretch mb-[20px]">
                <span className="text-[#FFFFFF] text-[48px] font-bold mb-[16px]">
                  Our project
                </span>
                <span className="text-[#FFFFFF] text-[20px] mb-[16px]">
                  Data science
                </span>
                <span className="text-[#FFFFFF] text-[16px]">
                  Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries,
                </span>
              </div>
              <div className="flex items-start">
                <button
                  className="flex flex-col shrink-0 items-start bg-[#FFFFFF] text-left pt-[8px] pb-[8px] pl-[16px] pr-[16px] mr-[12px] rounded-[6px] border-[1px] border-solid border-[#E2E8F0]"
                  onClick={handleClick}
                >
                  <span className="text-[#0F172A] text-[16px] font-bold">
                    Getting Started
                  </span>
                </button>
                <button
                  className="flex flex-col shrink-0 items-start bg-transparent text-left pt-[8px] pb-[8px] pl-[16px] pr-[16px] rounded-[6px] border-[1px] border-solid border-[#E2E8F0]"
                  onClick={handleClick} // Gắn sự kiện handleClick
                >
                  <span className="text-[#E2E8F0] text-[16px] font-bold">
                    Getting Started
                  </span>
                </button>
              </div>
            </div>
            <div className="flex flex-col shrink-0 items-start">
              <img
                src="https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/8cbee835-8b9c-426a-b33d-2432a4d55eff"
                className="w-[288px] h-[288px] object-fill"
                alt="image1"
              />
              <img
                src="https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/0cf5f1ff-4293-44d4-9e91-4833f6a5ea3c"
                className="w-[288px] h-[288px] ml-[94px] object-fill"
                alt="image2"
              />
            </div>
          </div>
          <button
            className="flex flex-col items-center self-stretch bg-[#FFFFFF] text-left pt-[17px] pb-[17px] rounded-tl-[16px] rounded-tr-[16px] border-0"
            onClick={handleClick} // Gắn sự kiện handleClick
          >
            <span className="text-[#0B0B0B] text-[14px] font-bold">
              rt-data-science 2025
            </span>
          </button>
        </div>
      </div>
    </>
  );
}
