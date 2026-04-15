import axios from 'axios';

const API_URL = 'http://localhost:8007';

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
  best_model_name: string;
  final_prediction: number;
  final_probability: number;
  final_risk_level: string;
  all_predictions: SingleModelPrediction[];
}

export interface TrainResponse {
  status: string;
  best_model: string;
  models_trained: string[];
  evaluation_results: Record<string, Record<string, number>>;
  plots_generated: string[];
}

export interface MetricsResponse {
  results: Record<string, Record<string, number>>;
  best_model: string;
  feature_selection: Record<string, string[]>;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  preprocessor_loaded: boolean;
  model_name: string | null;
}

class ApiService {
  private client = axios.create({
    baseURL: API_URL,
    timeout: 600000, // 10 min timeout for training
    headers: { 'Content-Type': 'application/json' },
  });

  async healthCheck(): Promise<HealthResponse> {
    const res = await this.client.get('/health');
    return res.data;
  }

  async predict(data: PatientData): Promise<PredictionResponse> {
    const res = await this.client.post<PredictionResponse>('/predict', data);
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

  async getPlotsList(): Promise<string[]> {
    const res = await this.client.get<{ plots: string[] }>('/plots-list');
    return res.data.plots;
  }

  getPlotUrl(filename: string): string {
    return `${API_URL}/plots/${filename}`;
  }
}

export const apiService = new ApiService();