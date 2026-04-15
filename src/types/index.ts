

export interface PatientData {
  age: number;
  bp: number;
  sg: number;
  al: number;
  su: number;
  rbc: 'normal' | 'abnormal';
  pc: 'normal' | 'abnormal';
  pcc: 'notpresent' | 'present';
  ba: 'notpresent' | 'present';
  bgr: number;
  bu: number;
  sc: number;
  sod: number;
  pot: number;
  hemo: number;
  pcv: number;
  wc: number;
  rc: number;
  htn: 'yes' | 'no';
  dm: 'yes' | 'no';
  cad: 'yes' | 'no';
  appet: 'good' | 'poor';
  pe: 'yes' | 'no';
  ane: 'yes' | 'no';
}

export interface PredictionResponse {
  prediction: 0 | 1;
  probability: number;
  risk_level: 'Low' | 'Medium' | 'High';
  confidence: 'Low' | 'Medium' | 'High';
}

export interface FormFieldConfig {
  name: keyof PatientData;
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