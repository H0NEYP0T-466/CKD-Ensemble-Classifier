import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { apiService } from '../services/api';
import type { PatientData, PredictionResponse } from '../services/api';
import type { FormFieldConfig } from '../types';
import FormField from '../components/FormField';
import RiskIndicator from '../components/RiskIndicator';
import LoadingSpinner from '../components/LoadingSpinner';

const formFields: FormFieldConfig[] = [
  // Basic Metrics
  { name: 'age', label: 'Age (years)', type: 'number', required: true, min: 0, max: 120, step: 1 },
  { name: 'bp', label: 'Blood Pressure (mm/Hg)', type: 'number', required: true, min: 0, max: 200, step: 1 },

  // Nominal that look numeric (per UCI spec)
  {
    name: 'sg', label: 'Specific Gravity', type: 'select', required: true,
    options: [
      { value: '1.005', label: '1.005' }, { value: '1.010', label: '1.010' },
      { value: '1.015', label: '1.015' }, { value: '1.020', label: '1.020' },
      { value: '1.025', label: '1.025' },
    ],
  },
  {
    name: 'al', label: 'Albumin', type: 'select', required: true,
    options: [
      { value: '0', label: '0' }, { value: '1', label: '1' }, { value: '2', label: '2' },
      { value: '3', label: '3' }, { value: '4', label: '4' }, { value: '5', label: '5' },
    ],
  },
  {
    name: 'su', label: 'Sugar', type: 'select', required: true,
    options: [
      { value: '0', label: '0' }, { value: '1', label: '1' }, { value: '2', label: '2' },
      { value: '3', label: '3' }, { value: '4', label: '4' }, { value: '5', label: '5' },
    ],
  },

  // Blood Tests (numerical)
  { name: 'bgr', label: 'Blood Glucose Random (mg/dl)', type: 'number', required: true, min: 0, max: 600, step: 1 },
  { name: 'bu', label: 'Blood Urea (mg/dl)', type: 'number', required: true, min: 0, max: 500, step: 0.1 },
  { name: 'sc', label: 'Serum Creatinine (mg/dl)', type: 'number', required: true, min: 0, max: 100, step: 0.1 },
  { name: 'sod', label: 'Sodium (mEq/L)', type: 'number', required: true, min: 0, max: 200, step: 1 },
  { name: 'pot', label: 'Potassium (mEq/L)', type: 'number', required: true, min: 0, max: 100, step: 0.1 },
  { name: 'hemo', label: 'Hemoglobin (gms)', type: 'number', required: true, min: 0, max: 20, step: 0.1 },

  // Cell Counts (numerical)
  { name: 'pcv', label: 'Packed Cell Volume', type: 'number', required: true, min: 0, max: 100, step: 1 },
  { name: 'wbcc', label: 'White Blood Cell Count', type: 'number', required: true, min: 0, max: 30000, step: 100 },
  { name: 'rbcc', label: 'Red Blood Cell Count (M/cmm)', type: 'number', required: true, min: 0, max: 10, step: 0.1 },

  // Categorical
  { name: 'rbc', label: 'Red Blood Cells', type: 'select', required: true,
    options: [{ value: 'normal', label: 'Normal' }, { value: 'abnormal', label: 'Abnormal' }] },
  { name: 'pc', label: 'Pus Cells', type: 'select', required: true,
    options: [{ value: 'normal', label: 'Normal' }, { value: 'abnormal', label: 'Abnormal' }] },
  { name: 'pcc', label: 'Pus Cell Clumps', type: 'select', required: true,
    options: [{ value: 'notpresent', label: 'Not Present' }, { value: 'present', label: 'Present' }] },
  { name: 'ba', label: 'Bacteria', type: 'select', required: true,
    options: [{ value: 'notpresent', label: 'Not Present' }, { value: 'present', label: 'Present' }] },
  { name: 'htn', label: 'Hypertension', type: 'select', required: true,
    options: [{ value: 'no', label: 'No' }, { value: 'yes', label: 'Yes' }] },
  { name: 'dm', label: 'Diabetes Mellitus', type: 'select', required: true,
    options: [{ value: 'no', label: 'No' }, { value: 'yes', label: 'Yes' }] },
  { name: 'cad', label: 'Coronary Artery Disease', type: 'select', required: true,
    options: [{ value: 'no', label: 'No' }, { value: 'yes', label: 'Yes' }] },
  { name: 'appet', label: 'Appetite', type: 'select', required: true,
    options: [{ value: 'good', label: 'Good' }, { value: 'poor', label: 'Poor' }] },
  { name: 'pe', label: 'Pedal Edema', type: 'select', required: true,
    options: [{ value: 'no', label: 'No' }, { value: 'yes', label: 'Yes' }] },
  { name: 'ane', label: 'Anemia', type: 'select', required: true,
    options: [{ value: 'no', label: 'No' }, { value: 'yes', label: 'Yes' }] },
];

const initialFormData: Record<string, string> = {
  age: '', bp: '', sg: '1.020', al: '0', su: '0',
  rbc: 'normal', pc: 'normal', pcc: 'notpresent', ba: 'notpresent',
  bgr: '', bu: '', sc: '', sod: '', pot: '', hemo: '', pcv: '', wbcc: '', rbcc: '',
  htn: 'no', dm: 'no', cad: 'no', appet: 'good', pe: 'no', ane: 'no',
};

const PredictionPage: React.FC = () => {
  const [formData, setFormData] = useState<Record<string, string>>({ ...initialFormData });
  const [showResults, setShowResults] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

  const mutation = useMutation({
    mutationFn: (data: PatientData) => apiService.predict(data),
    onSuccess: (data) => {
      setPrediction(data);
      setShowResults(true);
    },
    onError: (error: any) => {
      alert(`Prediction failed: ${error.response?.data?.detail || error.message}`);
    },
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setShowResults(false);

    const numericFields = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc'];
    const patientData: PatientData = {} as PatientData;

    for (const [key, value] of Object.entries(formData)) {
      if (numericFields.includes(key)) {
        (patientData as any)[key] = value ? parseFloat(value) : null;
      } else {
        (patientData as any)[key] = value || null;
      }
    }

    mutation.mutate(patientData);
  };

  const handleReset = () => {
    setFormData({ ...initialFormData });
    setShowResults(false);
    setPrediction(null);
  };

  const numericBasicFields = ['age', 'bp'];
  const nominalFields = ['sg', 'al', 'su'];
  const bloodFields = ['bgr', 'bu', 'sc', 'sod', 'pot', 'hemo'];
  const cellFields = ['pcv', 'wbcc', 'rbcc'];
  const categoricalFields = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'];

  const renderFieldGroup = (names: string[]) =>
    formFields
      .filter(f => names.includes(f.name))
      .map(field => (
        <FormField key={field.name} field={field} value={formData[field.name]} onChange={handleInputChange} />
      ));

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>🔬 CKD Risk Prediction</h1>
        <p>Enter patient clinical parameters to predict Chronic Kidney Disease risk</p>
      </div>

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            <span className="icon">📋</span> Patient Data Input
          </h2>
          {mutation.isPending && <LoadingSpinner />}
        </div>

        <form onSubmit={handleSubmit} id="prediction-form">
          <div className="form-section">
            <h3 className="section-title">Basic Metrics</h3>
            <div className="form-grid">{renderFieldGroup(numericBasicFields)}</div>
          </div>

          <div className="form-section">
            <h3 className="section-title">Urinalysis (Nominal)</h3>
            <div className="form-grid">{renderFieldGroup(nominalFields)}</div>
          </div>

          <div className="form-section">
            <h3 className="section-title">Blood Test Results</h3>
            <div className="form-grid">{renderFieldGroup(bloodFields)}</div>
          </div>

          <div className="form-section">
            <h3 className="section-title">Cell Counts</h3>
            <div className="form-grid">{renderFieldGroup(cellFields)}</div>
          </div>

          <div className="form-section">
            <h3 className="section-title">Clinical Observations</h3>
            <div className="form-grid">{renderFieldGroup(categoricalFields)}</div>
          </div>

          <div className="form-actions">
            <button type="submit" className="btn btn-primary" disabled={mutation.isPending} id="predict-btn">
              {mutation.isPending ? '⏳ Predicting...' : '🔬 Predict CKD Risk'}
            </button>
            <button type="button" className="btn btn-secondary" onClick={handleReset} id="reset-btn">
              🔄 Reset Form
            </button>
          </div>
        </form>
      </div>

      {/* Results */}
      <div className={`results-panel ${showResults ? 'visible' : ''}`}>
        {/* Best Model Final Verdict */}
        <div className="card" style={{ marginBottom: 'var(--space-lg)' }}>
          <div className="card-header">
            <h2 className="card-title">
              <span className="icon">📊</span> Final Verdict
            </h2>
            {prediction && (
              <span className="badge badge-info">
                Best Model: {prediction.best_model_name?.replace(/_/g, ' ')}
              </span>
            )}
          </div>

          {prediction && (
            <>
              <div className="result-grid">
                <div className="result-item">
                  <span className="result-label">Diagnosis</span>
                  <span className={`result-value ${prediction.final_prediction === 1 ? 'ckd' : 'no-ckd'}`}>
                    {prediction.final_prediction === 1 ? '⚠️ CKD Detected' : '✅ No CKD'}
                  </span>
                </div>
                <div className="result-item">
                  <span className="result-label">Probability</span>
                  <span className="result-value probability">
                    {(prediction.final_probability * 100).toFixed(2)}%
                  </span>
                  <div className="probability-bar">
                    <div className="probability-fill" style={{ width: `${prediction.final_probability * 100}%` }} />
                  </div>
                </div>
              </div>
              <RiskIndicator
                riskLevel={prediction.final_risk_level as 'Low' | 'Medium' | 'High'}
                confidence={
                  prediction.all_predictions?.find(p => p.model_name === prediction.best_model_name)?.confidence as 'Low' | 'Medium' | 'High' || 'High'
                }
              />
            </>
          )}
        </div>

        {/* All Model Predictions Table */}
        {prediction && prediction.all_predictions && prediction.all_predictions.length > 0 && (
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">
                <span className="icon">🧬</span> All Model Predictions
              </h2>
              <span className="badge badge-success">
                {prediction.all_predictions.length} models
              </span>
            </div>

            <div className="metrics-table-container">
              <table className="metrics-table" id="all-models-prediction-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Prediction</th>
                    <th>Probability</th>
                    <th>Risk Level</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {prediction.all_predictions.map(p => (
                    <tr key={p.model_name} className={p.model_name === prediction.best_model_name ? 'best-row' : ''}>
                      <td className="model-name">
                        {p.model_name.replace(/_/g, ' ')}
                        {p.model_name === prediction.best_model_name && (
                          <span className="badge badge-success" style={{ marginLeft: 8 }}>Best</span>
                        )}
                      </td>
                      <td>
                        <span className={p.prediction === 1 ? 'text-danger' : 'text-success'} style={{ fontWeight: 700 }}>
                          {p.prediction === 1 ? '⚠️ CKD' : '✅ No CKD'}
                        </span>
                      </td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span style={{ fontFamily: 'var(--font-mono)', minWidth: 60 }}>
                            {(p.probability * 100).toFixed(2)}%
                          </span>
                          <div className="probability-bar" style={{ flex: 1, maxWidth: 120 }}>
                            <div className="probability-fill" style={{ width: `${p.probability * 100}%` }} />
                          </div>
                        </div>
                      </td>
                      <td>
                        <span className={`badge ${
                          p.risk_level === 'High' ? 'badge-danger' :
                          p.risk_level === 'Medium' ? 'badge-warning' : 'badge-success'
                        }`}>
                          {p.risk_level}
                        </span>
                      </td>
                      <td>
                        <span className={`badge ${
                          p.confidence === 'High' ? 'badge-success' :
                          p.confidence === 'Medium' ? 'badge-warning' : 'badge-danger'
                        }`}>
                          {p.confidence}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionPage;
