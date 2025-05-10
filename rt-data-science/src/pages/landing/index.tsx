import { useNavigate } from "react-router-dom";

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <>
      <div className="flex flex-col bg-black">
        <div className="flex items-center px-36 py-12">
          <span className="flex-1 text-white text-xl">Rotten Tomatoes</span>
          <div className="flex items-center">
            <span className="text-white mr-6">Option</span>
            <span className="text-white mr-6">Option</span>
            <span className="text-white ">Option</span>
          </div>
        </div>
        <div className="flex items-center px-16 py-12 mx-20 mb-32">
          <div className="flex-1 flex flex-col items-start mr-3">
            <div className="flex flex-col items-start mb-5">
              <span className="text-white text-5xl font-bold mb-4">
                Our project
              </span>
              <span className="text-white text-xl mb-4">Data science</span>
              <span className="text-white text-justify ">
                Lorem Ipsum is simply dummy text of the printing and typesetting
                industry. Lorem Ipsum has been the industry's standard dummy
                text ever since the 1500s, when an unknown printer took a galley
                of type and scrambled it to make a type specimen book. It has
                survived not only five centuries,
              </span>
            </div>
            <div className="flex items-start">
              <button
                className="bg-white text-left py-2 px-4 mr-3 rounded-md border border-gray-200"
                onClick={() => navigate("/home")}
              >
                <span className="text-gray-900 font-bold">
                  Getting Started
                </span>
              </button>
              <button
                className="bg-transparent text-left py-2 px-4 rounded-md border border-gray-200"
                onClick={() => navigate("/home")}
              >
                <span className="text-gray-200 font-bold">
                  Getting Started
                </span>
              </button>
            </div>
          </div>
          <div className="flex flex-col items-start ml-75 mr-40">
            <img
              src="https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/8cbee835-8b9c-426a-b33d-2432a4d55eff"
              className="w-72 h-72 object-fill"
              alt="image1"
            />
            <img
              src="https://figma-alpha-api.s3.us-west-2.amazonaws.com/images/0cf5f1ff-4293-44d4-9e91-4833f6a5ea3c"
              className="w-72 h-72 ml-24 object-fill"
              alt="image2"
            />
          </div>
        </div>
        <div className="flex justify-center bg-white py-4 rounded-t-2xl border-0 text-sm font-bold">
          rt-data-science 2025
        </div>
      </div>
    </>
  );
}
