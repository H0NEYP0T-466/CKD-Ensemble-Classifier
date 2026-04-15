import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import PredictionPage from './pages/PredictionPage';
import AnalyticsPage from './pages/AnalyticsPage';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Navbar />
        <Routes>
          <Route path="/" element={<PredictionPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
        </Routes>
        <footer className="app-footer">
          <p>
            Based on Rahman et al., 2024 — <em>Machine Learning models for chronic kidney disease diagnosis and prediction</em>
          </p>
        </footer>
      </div>
    </BrowserRouter>
  );
}

export default App;