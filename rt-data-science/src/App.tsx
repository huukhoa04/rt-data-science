import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import LandingPage from './pages/landing'
import HomePage from './pages/home'
import RecommendPage from './pages/recommend'
import MovieDetailPage from './pages/movieDetail';


function App() {

  return (
    <>
      <Router>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/home" element={<HomePage />} />
          <Route path="/recommend" element={<RecommendPage />} />
          <Route path="/movie/:slug" element={<MovieDetailPage />} />
        </Routes>
      </Router>
    </>
  )
}

export default App
