

import axios from 'axios';

const API_URL = 'http://localhost:8000';

export interface PatientData {
  age: number;
  bp: number;
  sg: number;
  al: number;
  su: number;
  rbc: string;
  pc: string;
  pcc: string;
  ba: string;
  bgr: number;
  bu: number;
  sc: number;
  sod: number;
  pot: number;
  hemo: number;
  pcv: number;
  wc: number;
  rc: number;
  htn: string;
  dm: string;
  cad: string;
  appet: string;
  pe: string;
  ane: string;
}

export interface PredictionResponse {
  prediction: number;
  probability: number;
  risk_level: string;
  confidence: string;
}

export interface BatchPredictionResponse extends PredictionResponse {
  id: number;
}

class ApiService {
  private client = axios.create({
    baseURL: API_URL,
    timeout: 10000,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  async healthCheck(): Promise<{ status: string; model_loaded: boolean }> {
    const response = await this.client.get('/health');
    return response.data;
  }

  async predict(patientData: PatientData): Promise<PredictionResponse> {
    try {
      const response = await this.client.post<PredictionResponse>('/predict', patientData);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw new Error(error.response.data.detail || 'Prediction failed');
      }
      throw error;
    }
  }

  async predictBatch(patientDataList: PatientData[]): Promise<BatchPredictionResponse[]> {
    try {
      const response = await this.client.post<BatchPredictionResponse[]>('/predict/batch', patientDataList);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw new Error(error.response.data.detail || 'Batch prediction failed');
      }
      throw error;
    }
  }

  async getFeatureNames(): Promise<{ feature_names: string[] }> {
    const response = await this.client.get('/features');
    return response.data;
  }
}

export const apiService = new ApiService();