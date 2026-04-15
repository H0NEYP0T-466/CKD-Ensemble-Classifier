import React from 'react';

interface MetricsData {
  [modelName: string]: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    auc_roc: number;
  };
}

interface MetricsTableProps {
  data: MetricsData;
  bestModel: string;
}

const MetricsTable: React.FC<MetricsTableProps> = ({ data, bestModel }) => {
  const models = Object.keys(data);
  const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'];

  // Find best value for each metric (for highlighting)
  const bestValues: Record<string, number> = {};
  metrics.forEach(metric => {
    bestValues[metric] = Math.max(
      ...models.map(m => data[m][metric as keyof typeof data[typeof m]] || 0)
    );
  });

  const formatMetric = (name: string) => {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  };

  return (
    <div className="metrics-table-container">
      <table className="metrics-table" id="metrics-comparison-table">
        <thead>
          <tr>
            <th>Model</th>
            {metrics.map(m => (
              <th key={m}>{formatMetric(m)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map(model => (
            <tr key={model} className={model === bestModel ? 'best-row' : ''}>
              <td className="model-name">
                {model.replace(/_/g, ' ')}
                {model === bestModel && (
                  <span className="badge badge-success" style={{ marginLeft: 8 }}>Best</span>
                )}
              </td>
              {metrics.map(metric => {
                const value = data[model][metric as keyof typeof data[typeof model]] || 0;
                const isBest = value === bestValues[metric] && value > 0;
                return (
                  <td key={metric} className={isBest ? 'best-value' : ''}>
                    {(value * 100).toFixed(2)}%
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default MetricsTable;
