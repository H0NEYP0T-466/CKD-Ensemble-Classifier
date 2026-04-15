import axios from 'axios';

const API_URL = 'http://localhost:8007';

export type Variant = 'all_features' | 'rfe' | 'boruta';

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

export interface SingleModelPrediction {
  model_name: string;
  prediction: number;
  probability: number;
  risk_level: string;
  confidence: string;
}

export interface PredictionResponse {
  variant: string;
  n_features: number;
  best_model_name: string;
  final_prediction: number;
  final_probability: number;
  final_risk_level: string;
  all_predictions: SingleModelPrediction[];
}

export interface TrainResponse {
  status: string;
  variants_trained: string[];
  total_models: number;
  variant_results: Record<string, {
    n_features: number;
    models_trained: string[];
    best_model: string;
  }>;
}

export interface VariantMetrics {
  variant: string;
  n_features: number;
  features: string[];
  results: Record<string, Record<string, number>>;
  best_model: string;
}

export interface MetricsResponse {
  variants: Record<string, {
    features: string[];
    n_features: number;
    models_trained: string[];
    evaluation_results: Record<string, Record<string, number>>;
    best_model: string;
  }>;
  feature_selection: Record<string, string[]>;
}

export interface HealthResponse {
  status: string;
  variants_loaded: string[];
  total_models: number;
  preprocessor_loaded: boolean;
}

export interface PlotsListResponse {
  plots: Record<string, string[]>;
}

class ApiService {
  private client = axios.create({
    baseURL: API_URL,
    timeout: 600000,
    headers: { 'Content-Type': 'application/json' },
  });

  async healthCheck(): Promise<HealthResponse> {
    const res = await this.client.get('/health');
    return res.data;
  }

  async predict(variant: Variant, data: PatientData): Promise<PredictionResponse> {
    const res = await this.client.post<PredictionResponse>(`/predict/${variant}`, data);
    return res.data;
  }

  async train(): Promise<TrainResponse> {
    const res = await this.client.post<TrainResponse>('/train');
    return res.data;
  }

  async getMetrics(): Promise<MetricsResponse> {
    const res = await this.client.get<MetricsResponse>('/metrics');
    return res.data;
  }

  async getVariantMetrics(variant: Variant): Promise<VariantMetrics> {
    const res = await this.client.get<VariantMetrics>(`/metrics/${variant}`);
    return res.data;
  }

  async getPlotsList(): Promise<PlotsListResponse> {
    const res = await this.client.get<PlotsListResponse>('/plots-list');
    return res.data;
  }

  getPlotUrl(filename: string, variant?: string): string {
    if (variant) {
      return `${API_URL}/plots/${variant}/${filename}`;
    }
    return `${API_URL}/plots/${filename}`;
  }
}

export const apiService = new ApiService();