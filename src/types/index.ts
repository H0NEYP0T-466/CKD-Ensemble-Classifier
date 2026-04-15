export interface PatientData {
  age: number | null;
  bp: number | null;
  sg: string;
  al: string;
  su: string;
  rbc: string;
  pc: string;
  pcc: string;
  ba: string;
  bgr: number | null;
  bu: number | null;
  sc: number | null;
  sod: number | null;
  pot: number | null;
  hemo: number | null;
  pcv: number | null;
  wbcc: number | null;
  rbcc: number | null;
  htn: string;
  dm: string;
  cad: string;
  appet: string;
  pe: string;
  ane: string;
}

export interface PredictionResponse {
  prediction: 0 | 1;
  probability: number;
  risk_level: 'Low' | 'Medium' | 'High';
  confidence: 'Low' | 'Medium' | 'High';
  model_name: string;
}

export interface FormFieldConfig {
  name: string;
  label: string;
  type: 'number' | 'select';
  required?: boolean;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
}

export interface RiskIndicatorProps {
  riskLevel: 'Low' | 'Medium' | 'High';
  confidence: 'Low' | 'Medium' | 'High';
}

export interface MetricsData {
  [modelName: string]: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    auc_roc: number;
  };
}