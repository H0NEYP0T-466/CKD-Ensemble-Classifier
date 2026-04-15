import React from 'react';
import type { RiskIndicatorProps } from '../types';

const RiskIndicator: React.FC<RiskIndicatorProps> = ({ riskLevel, confidence }) => {
  const getLedColor = () => {
    switch (riskLevel) {
      case 'Low':
        return 'terminal-green';
      case 'Medium':
        return 'warning';
      case 'High':
        return 'terminal-red';
      default:
        return 'terminal-green';
    }
  };

  const getConfidenceColor = () => {
    switch (confidence) {
      case 'High':
        return 'terminal-green';
      case 'Medium':
        return 'warning';
      case 'Low':
        return 'terminal-red';
      default:
        return 'terminal-green';
    }
  };

  return (
    <div className="risk-indicators">
      <div className="risk-indicator">
        <div className={`risk-led ${getLedColor()}`}></div>
        <span className="risk-text">Risk Level: {riskLevel}</span>
      </div>
      <div className="risk-indicator">
        <div className={`risk-led ${getConfidenceColor()}`}></div>
        <span className="risk-text">Confidence: {confidence}</span>
      </div>
    </div>
  );
};

export default RiskIndicator;