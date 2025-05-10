import { useNavigate } from "react-router-dom";

interface HeaderProps {
  onSearch: (query: string) => void;
}

export default function Header({ onSearch }: HeaderProps) {
  const navigate = useNavigate();

  return (
    <>
      <div className="flex flex-col px-50 py-12 gap-8">
        <div className="flex items-center space-x-8">
          <div className="text-2xl font-bold whitespace-nowrap">
            <button onClick={() => navigate("/home")}>Rotten Tomato</button>
          </div>
          <input
            type="text"
            placeholder="Search"
            className="flex-1 text-gray-800 bg-white py-3 px-4 rounded-full border border-gray-300"
            onChange={(e) => onSearch(e.target.value)}
          />
          <div className="flex space-x-6">
            <button onClick={() => alert("Pressed!")}>Option</button>
            <button onClick={() => alert("Pressed!")}>Option</button>
            <button onClick={() => alert("Pressed!")}>Option</button>
          </div>
        </div>
        <div className="flex justify-between px-50">
          <button
            className="bg-white py-2 px-12 rounded-md border border-black shadow-md"
            onClick={() => navigate("/recommend")}
          >
            <span className="font-bold">Gợi ý phim</span>
          </button>
          <button
            className="bg-white py-2 px-14 rounded-md border border-black shadow-md"
            onClick={() => alert("Pressed!")}
          >
            <span className="font-bold">Phổ biến</span>
          </button>
          <button
            className="bg-white py-2 px-12 rounded-md border border-black shadow-md"
            onClick={() => alert("Pressed!")}
          >
            <span className="font-bold">Mới ra mắt</span>
          </button>
        </div>
      </div>
    </>
  );
}
