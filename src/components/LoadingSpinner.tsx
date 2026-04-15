import React from 'react';

const LoadingSpinner: React.FC = () => {
  return (
    <div className="loading-indicator">
      <div className="spinner"></div>
      <span>Processing prediction...</span>
    </div>
  );
};

export default LoadingSpinner;