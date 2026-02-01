import axios from 'axios';

const API_BASE_URL = 'https://ml-churn-1.onrender.com';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const predictSingle = async (customerData) => {
  const response = await api.post('/predict', customerData);
  return response.data;
};

export const predictBatch = async (customers) => {
  const response = await api.post('/predict/batch', { customers });
  return response.data;
};

export const predictFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/predict/file', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const validateFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/validate/file', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const downloadFile = (filename) => {
  return `${API_BASE_URL}/downloads/${filename}`;
};

export default api;
