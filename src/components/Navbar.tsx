import React from 'react';
import { NavLink } from 'react-router-dom';

const Navbar: React.FC = () => {
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <div className="navbar-brand">
          <div className="brand-icon">🧬</div>
          <span className="brand-ckd">CKD</span>
          <span className="brand-sep">|</span>
          <span className="brand-text">Ensemble Classifier</span>
        </div>
        <div className="navbar-links">
          <NavLink
            to="/"
            end
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            🔬 Prediction
          </NavLink>
          <NavLink
            to="/analytics"
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            📊 Analytics
          </NavLink>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
