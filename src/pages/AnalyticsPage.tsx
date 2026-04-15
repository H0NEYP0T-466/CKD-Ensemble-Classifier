import React, { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import type { MetricsResponse, TrainResponse } from '../services/api';
import MetricsTable from '../components/MetricsTable';

const AnalyticsPage: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [plots, setPlots] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [trainResult, setTrainResult] = useState<TrainResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [metricsData, plotsList] = await Promise.all([
        apiService.getMetrics().catch(() => null),
        apiService.getPlotsList().catch(() => []),
      ]);
      setMetrics(metricsData);
      setPlots(plotsList);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleTrain = async () => {
    if (!confirm('This will run the full ML training pipeline. It may take several minutes. Continue?')) {
      return;
    }
    setTraining(true);
    setError(null);
    try {
      const result = await apiService.train();
      setTrainResult(result);
      // Refresh data
      await fetchData();
    } catch (err: any) {
      setError(`Training failed: ${err.response?.data?.detail || err.message}`);
    } finally {
      setTraining(false);
    }
  };

  // Categorize plots
  const kdePlots = plots.filter(p => p.includes('kde'));
  const comparisonPlots = plots.filter(p => p.includes('comparison'));
  const cmPlots = plots.filter(p => p.startsWith('cm_'));
  const rocPlots = plots.filter(p => p.includes('roc'));

  const renderPlotLabel = (filename: string) => {
    return filename
      .replace('.png', '')
      .replace(/^cm_/, 'Confusion Matrix: ')
      .replace(/^comparison_/, '')
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());
  };

  return (
    <div className="page-container">
      <div className="analytics-header">
        <div>
          <h1>📊 Model Analytics Dashboard</h1>
          <p className="subtitle">
            Evaluation metrics and visualizations from the ensemble classifier pipeline
          </p>
        </div>
        <button
          className={`btn ${training ? 'btn-secondary' : 'btn-primary'}`}
          onClick={handleTrain}
          disabled={training}
          id="train-btn"
        >
          {training ? (
            <>
              <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
              Training Pipeline...
            </>
          ) : (
            '🚀 Train Models'
          )}
        </button>
      </div>

      {error && (
        <div className="card" style={{ borderColor: 'var(--accent-danger)', marginBottom: 'var(--space-xl)' }}>
          <p style={{ color: 'var(--accent-danger)' }}>⚠️ {error}</p>
        </div>
      )}

      {trainResult && (
        <div className="card" style={{ borderColor: 'var(--accent-success)', marginBottom: 'var(--space-xl)' }}>
          <p style={{ color: 'var(--accent-success)', fontWeight: 600 }}>
            ✅ Training completed! Best model: {trainResult.best_model?.replace(/_/g, ' ')}
          </p>
          <p className="text-secondary" style={{ marginTop: 8, fontSize: '0.9rem' }}>
            {trainResult.models_trained?.length} models trained, {trainResult.plots_generated?.length} plots generated.
          </p>
        </div>
      )}

      {loading ? (
        <div className="empty-state">
          <div className="spinner" style={{ width: 40, height: 40, margin: '0 auto 16px' }} />
          <h3>Loading analytics...</h3>
        </div>
      ) : !metrics ? (
        <div className="empty-state">
          <div className="icon">🧪</div>
          <h3>No trained models found</h3>
          <p>Click "Train Models" above to run the ML pipeline and generate evaluation metrics and plots.</p>
        </div>
      ) : (
        <>
          {/* Metrics Table */}
          <div className="analytics-section">
            <h2 className="analytics-section-title">
              <span className="dot" /> Evaluation Metrics — All Ensemble Models
            </h2>
            <MetricsTable data={metrics.results} bestModel={metrics.best_model} />
          </div>

          {/* Feature Selection Info */}
          {metrics.feature_selection && metrics.feature_selection.rfe_features && (
            <div className="analytics-section">
              <h2 className="analytics-section-title">
                <span className="dot" /> Feature Selection Results
              </h2>
              <div className="card">
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-xl)' }}>
                  <div>
                    <h4 style={{ color: 'var(--accent-primary)', marginBottom: 'var(--space-sm)' }}>
                      RFE Features ({metrics.feature_selection.rfe_features?.length})
                    </h4>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {metrics.feature_selection.rfe_features?.map(f => (
                        <span key={f} className="badge badge-info">{f}</span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <h4 style={{ color: 'var(--accent-secondary)', marginBottom: 'var(--space-sm)' }}>
                      Boruta Features ({metrics.feature_selection.boruta_features?.length})
                    </h4>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {metrics.feature_selection.boruta_features?.map(f => (
                        <span key={f} className="badge badge-success">{f}</span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* KDE Distributions */}
          {kdePlots.length > 0 && (
            <div className="analytics-section">
              <h2 className="analytics-section-title">
                <span className="dot" /> Feature Distributions (KDE) — CKD vs NOTCKD
              </h2>
              <div className="plots-grid">
                {kdePlots.map(plot => (
                  <div key={plot} className="plot-card plot-full-width">
                    <img src={apiService.getPlotUrl(plot)} alt={renderPlotLabel(plot)} loading="lazy" />
                    <div className="plot-label">{renderPlotLabel(plot)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Model Comparison Bar Charts */}
          {comparisonPlots.length > 0 && (
            <div className="analytics-section">
              <h2 className="analytics-section-title">
                <span className="dot" /> Model Performance Comparison
              </h2>
              <div className="plots-grid">
                {comparisonPlots.map(plot => (
                  <div key={plot} className={`plot-card ${plot.includes('all_metrics') ? 'plot-full-width' : ''}`}>
                    <img src={apiService.getPlotUrl(plot)} alt={renderPlotLabel(plot)} loading="lazy" />
                    <div className="plot-label">{renderPlotLabel(plot)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ROC Curves */}
          {rocPlots.length > 0 && (
            <div className="analytics-section">
              <h2 className="analytics-section-title">
                <span className="dot" /> ROC Curves
              </h2>
              <div className="plots-grid">
                {rocPlots.map(plot => (
                  <div key={plot} className="plot-card plot-full-width">
                    <img src={apiService.getPlotUrl(plot)} alt={renderPlotLabel(plot)} loading="lazy" />
                    <div className="plot-label">{renderPlotLabel(plot)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Confusion Matrices */}
          {cmPlots.length > 0 && (
            <div className="analytics-section">
              <h2 className="analytics-section-title">
                <span className="dot" /> Confusion Matrices
              </h2>
              <div className="plots-grid">
                {cmPlots.map(plot => (
                  <div key={plot} className="plot-card">
                    <img src={apiService.getPlotUrl(plot)} alt={renderPlotLabel(plot)} loading="lazy" />
                    <div className="plot-label">{renderPlotLabel(plot)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default AnalyticsPage;
