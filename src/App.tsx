import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { apiService } from './services/api';
import type { PatientData, PredictionResponse } from './services/api';
import type { FormFieldConfig } from './types';
import FormField from './components/FormField';
import RiskIndicator from './components/RiskIndicator';
import LoadingSpinner from './components/LoadingSpinner';
import './App.css';

const formFields: FormFieldConfig[] = [
  // Basic Metrics
  { name: 'age', label: 'Age (years)', type: 'number', required: true, min: 0, max: 120, step: 0.1 },
  { name: 'bp', label: 'Blood Pressure (mm/Hg)', type: 'number', required: true, min: 0, max: 200, step: 0.1 },
  { name: 'sg', label: 'Specific Gravity', type: 'number', required: true, min: 0, max: 2, step: 0.01 },
  { name: 'al', label: 'Albumin', type: 'number', required: true, min: 0, max: 5, step: 0.1 },
  { name: 'su', label: 'Sugar', type: 'number', required: true, min: 0, max: 5, step: 0.1 },

  // Blood Tests
  { name: 'bgr', label: 'Blood Glucose Random (mgs/dl)', type: 'number', required: true, min: 0, max: 500, step: 1 },
  { name: 'bu', label: 'Blood Urea (mgs/dl)', type: 'number', required: true, min: 0, max: 500, step: 1 },
  { name: 'sc', label: 'Serum Creatinine (mgs/dl)', type: 'number', required: true, min: 0, max: 100, step: 0.1 },
  { name: 'sod', label: 'Sodium (mEq/L)', type: 'number', required: true, min: 0, max: 200, step: 1 },
  { name: 'pot', label: 'Potassium (mEq/L)', type: 'number', required: true, min: 0, max: 100, step: 1 },
  { name: 'hemo', label: 'Hemoglobin (gms)', type: 'number', required: true, min: 0, max: 20, step: 0.1 },

  // Cell Counts
  { name: 'pcv', label: 'Packed Cell Volume', type: 'number', required: true, min: 0, max: 100, step: 1 },
  { name: 'wc', label: 'White Blood Cell Count', type: 'number', required: true, min: 0, max: 50000, step: 1 },
  { name: 'rc', label: 'Red Blood Cell Count', type: 'number', required: true, min: 0, max: 10, step: 0.1 },

  // Categorical - Normal/Abnormal
  {
    name: 'rbc',
    label: 'Red Blood Cells',
    type: 'select',
    required: true,
    options: [
      { value: 'normal', label: 'Normal' },
      { value: 'abnormal', label: 'Abnormal' },
    ],
  },
  {
    name: 'pc',
    label: 'Pus Cells',
    type: 'select',
    required: true,
    options: [
      { value: 'normal', label: 'Normal' },
      { value: 'abnormal', label: 'Abnormal' },
    ],
  },

  // Categorical - Present/Not Present
  {
    name: 'pcc',
    label: 'Pus Cell Clumps',
    type: 'select',
    required: true,
    options: [
      { value: 'notpresent', label: 'Not Present' },
      { value: 'present', label: 'Present' },
    ],
  },
  {
    name: 'ba',
    label: 'Bacteria',
    type: 'select',
    required: true,
    options: [
      { value: 'notpresent', label: 'Not Present' },
      { value: 'present', label: 'Present' },
    ],
  },

  // Categorical - Yes/No
  {
    name: 'htn',
    label: 'Hypertension',
    type: 'select',
    required: true,
    options: [
      { value: 'no', label: 'No' },
      { value: 'yes', label: 'Yes' },
    ],
  },
  {
    name: 'dm',
    label: 'Diabetes Mellitus',
    type: 'select',
    required: true,
    options: [
      { value: 'no', label: 'No' },
      { value: 'yes', label: 'Yes' },
    ],
  },
  {
    name: 'cad',
    label: 'Coronary Artery Disease',
    type: 'select',
    required: true,
    options: [
      { value: 'no', label: 'No' },
      { value: 'yes', label: 'Yes' },
    ],
  },
  {
    name: 'pe',
    label: 'Pedal Edema',
    type: 'select',
    required: true,
    options: [
      { value: 'no', label: 'No' },
      { value: 'yes', label: 'Yes' },
    ],
  },
  {
    name: 'ane',
    label: 'Anemia',
    type: 'select',
    required: true,
    options: [
      { value: 'no', label: 'No' },
      { value: 'yes', label: 'Yes' },
    ],
  },

  // Categorical - Good/Poor
  {
    name: 'appet',
    label: 'Appetite',
    type: 'select',
    required: true,
    options: [
      { value: 'good', label: 'Good' },
      { value: 'poor', label: 'Poor' },
    ],
  },
];

const App: React.FC = () => {
  const [formData, setFormData] = useState<Record<string, string>>({
    age: '', bp: '', sg: '', al: '', su: '', rbc: 'normal', pc: 'normal', pcc: 'notpresent',
    ba: 'notpresent', bgr: '', bu: '', sc: '', sod: '', pot: '', hemo: '', pcv: '', wc: '',
    rc: '', htn: 'no', dm: 'no', cad: 'no', appet: 'good', pe: 'no', ane: 'no'
  });

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
    }
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setShowResults(false);

    // Convert form data to PatientData format
    const patientData: PatientData = {
      age: parseFloat(formData.age),
      bp: parseFloat(formData.bp),
      sg: parseFloat(formData.sg),
      al: parseFloat(formData.al),
      su: parseFloat(formData.su),
      rbc: formData.rbc as 'normal' | 'abnormal',
      pc: formData.pc as 'normal' | 'abnormal',
      pcc: formData.pcc as 'notpresent' | 'present',
      ba: formData.ba as 'notpresent' | 'present',
      bgr: parseFloat(formData.bgr),
      bu: parseFloat(formData.bu),
      sc: parseFloat(formData.sc),
      sod: parseFloat(formData.sod),
      pot: parseFloat(formData.pot),
      hemo: parseFloat(formData.hemo),
      pcv: parseFloat(formData.pcv),
      wc: parseFloat(formData.wc),
      rc: parseFloat(formData.rc),
      htn: formData.htn as 'yes' | 'no',
      dm: formData.dm as 'yes' | 'no',
      cad: formData.cad as 'yes' | 'no',
      appet: formData.appet as 'good' | 'poor',
      pe: formData.pe as 'yes' | 'no',
      ane: formData.ane as 'yes' | 'no',
    };

    mutation.mutate(patientData);
  };

  const handleReset = () => {
    setFormData({
      age: '', bp: '', sg: '', al: '', su: '', rbc: 'normal', pc: 'normal', pcc: 'notpresent',
      ba: 'notpresent', bgr: '', bu: '', sc: '', sod: '', pot: '', hemo: '', pcv: '', wc: '',
      rc: '', htn: 'no', dm: 'no', cad: 'no', appet: 'good', pe: 'no', ane: 'no'
    });
    setShowResults(false);
    setPrediction(null);
  };

  return (
    <div className="app-container">
      <header className="cyber-header">
        <h1 className="cyber-title">
          <span className="neon-cyan">CKD</span>
          <span className="neon-purple">ENSEMBLE</span>
          <span className="neon-cyan">CLASSIFIER</span>
        </h1>
        <p className="cyber-subtitle">Chronic Kidney Disease Prediction System</p>
      </header>

      <main className="main-content">
        <section className="prediction-panel">
          <div className="panel-header">
            <h2 className="panel-title">Patient Data Input</h2>
            {mutation.isPending && <LoadingSpinner />}
          </div>

          <form className="patient-form" onSubmit={handleSubmit}>
            {/* Basic Metrics Section */}
            <div className="form-section">
              <h3 className="section-title">Basic Metrics</h3>
              <div className="form-grid">
                {formFields
                  .filter(field => ['age', 'bp', 'sg', 'al', 'su'].includes(field.name))
                  .map(field => (
                    <FormField
                      key={field.name}
                      field={field}
                      value={formData[field.name]}
                      onChange={handleInputChange}
                    />
                  ))}
              </div>
            </div>

            {/* Blood Test Results Section */}
            <div className="form-section">
              <h3 className="section-title">Blood Test Results</h3>
              <div className="form-grid">
                {formFields
                  .filter(field => ['bgr', 'bu', 'sc', 'sod', 'pot', 'hemo'].includes(field.name))
                  .map(field => (
                    <FormField
                      key={field.name}
                      field={field}
                      value={formData[field.name]}
                      onChange={handleInputChange}
                    />
                  ))}
              </div>
            </div>

            {/* Cell Counts Section */}
            <div className="form-section">
              <h3 className="section-title">Cell Counts</h3>
              <div className="form-grid">
                {formFields
                  .filter(field => ['pcv', 'wc', 'rc'].includes(field.name))
                  .map(field => (
                    <FormField
                      key={field.name}
                      field={field}
                      value={formData[field.name]}
                      onChange={handleInputChange}
                    />
                  ))}
              </div>
            </div>

            {/* Clinical Observations Section */}
            <div className="form-section">
              <h3 className="section-title">Clinical Observations</h3>
              <div className="form-grid">
                {formFields
                  .filter(field => !['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc'].includes(field.name))
                  .map(field => (
                    <FormField
                      key={field.name}
                      field={field}
                      value={formData[field.name]}
                      onChange={handleInputChange}
                    />
                  ))}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="form-actions">
              <button
                type="submit"
                className="cyber-button primary"
                disabled={mutation.isPending}
              >
                <span className="button-text">Predict CKD Risk</span>
                <span className="button-glow"></span>
              </button>
              <button
                type="button"
                className="cyber-button secondary"
                onClick={handleReset}
              >
                <span className="button-text">Reset Form</span>
              </button>
            </div>
          </form>
        </section>

        {/* Results Panel */}
        <section className={`results-panel ${showResults ? 'visible' : ''}`}>
          <div className="panel-header">
            <h2 className="panel-title">Prediction Results</h2>
          </div>
          <div className="results-content">
            {prediction && (
              <>
                <div className="result-item">
                  <span className="result-label">Prediction:</span>
                  <span className={`result-value ${prediction.prediction === 1 ? 'ckd' : 'no-ckd'}`}>
                    {prediction.prediction === 1 ? 'CKD Detected' : 'No CKD'}
                  </span>
                </div>
                <div className="result-item">
                  <span className="result-label">Probability:</span>
                  <span className="result-value probability">
                    {(prediction.probability * 100).toFixed(2)}%
                  </span>
                  <div className="probability-bar">
                    <div
                      className="probability-fill"
                      style={{ width: `${prediction.probability * 100}%` }}
                    ></div>
                  </div>
                </div>
                <RiskIndicator riskLevel={prediction.risk_level as 'Low' | 'Medium' | 'High'} confidence={prediction.confidence as 'Low' | 'Medium' | 'High'} />
              </>
            )}
          </div>
        </section>
      </main>

      <footer className="cyber-footer">
        <p>Based on Rahman et al., 2024 - Machine Learning models for chronic kidney disease diagnosis and prediction</p>
      </footer>
    </div>
  );
};

export default App;