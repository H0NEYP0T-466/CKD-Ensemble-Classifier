# 🩺 CKD Ensemble Classifier

<p align="center">

  <!-- Core -->
  ![GitHub License](https://img.shields.io/github/license/H0NEYP0T-466/CKD-Ensemble-Classifier?style=for-the-badge&color=brightgreen)
  ![GitHub Stars](https://img.shields.io/github/stars/H0NEYP0T-466/CKD-Ensemble-Classifier?style=for-the-badge&color=yellow)
  ![GitHub Forks](https://img.shields.io/github/forks/H0NEYP0T-466/CKD-Ensemble-Classifier?style=for-the-badge&color=blue)
  ![GitHub Issues](https://img.shields.io/github/issues/H0NEYP0T-466/CKD-Ensemble-Classifier?style=for-the-badge&color=red)
  ![GitHub Pull Requests](https://img.shields.io/github/issues-pr/H0NEYP0T-466/CKD-Ensemble-Classifier?style=for-the-badge&color=orange)
  ![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge)

  <!-- Activity -->
  ![Last Commit](https://img.shields.io/github/last-commit/H0NEYP0T-466/CKD-Ensemble-Classifier?style=for-the-badge&color=purple)
  ![Commit Activity](https://img.shields.io/github/commit-activity/m/H0NEYP0T-466/CKD-Ensemble-Classifier?style=for-the-badge&color=teal)
  ![Repo Size](https://img.shields.io/github/repo-size/H0NEYP0T-466/CKD-Ensemble-Classifier?style=for-the-badge&color=blueviolet)

  <!-- Languages -->
  ![Top Language](https://img.shields.io/github/languages/top/H0NEYP0T-466/CKD-Ensemble-Classifier?style=for-the-badge&color=critical)
  ![Languages Count](https://img.shields.io/github/languages/count/H0NEYP0T-466/CKD-Ensemble-Classifier?style=for-the-badge&color=success)

  <!-- Community -->
  ![Discussions](https://img.shields.io/github/discussions/H0NEYP0T-466/CKD-Ensemble-Classifier?style=for-the-badge&color=blue)
  ![Documentation](https://img.shields.io/badge/Docs-Available-green?style=for-the-badge&logo=readthedocs&logoColor=white)
  ![Open Source Love](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red?style=for-the-badge)

</p>

> 🔬 A machine learning system for **Chronic Kidney Disease (CKD) prediction** using ensemble methods, based on Rahman et al., 2024 methodology. Features a FastAPI backend with 24 trained models and a React + TypeScript frontend.

🌐 **Frontend:** [ckd-ensemble-classifier.vercel.app](https://ckd-ensemble-classifier.vercel.app/) (Deployed on Vercel)
⚙️ **Backend:** Self-hosted FastAPI service

---

## 🔗 Quick Links

| Link | Description |
|------|-------------|
| 🌐 [Live Demo](https://ckd-ensemble-classifier.vercel.app/) | Frontend deployed on Vercel |
| 📖 [Documentation](#-table-of-contents) | Full project documentation |
| 🐛 [Issues](https://github.com/H0NEYP0T-466/CKD-Ensemble-Classifier/issues) | Report bugs or request features |
| 🤝 [Contributing](CONTRIBUTING.md) | How to contribute |

---

## 📑 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)
- [Dependencies & Packages](#-dependencies--packages)
- [Folder Structure](#-folder-structure)
- [Model Performance & Charts](#-model-performance--charts)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)
- [Security](#-security)
- [Code of Conduct](#-code-of-conduct)

---

## ✨ Features

- 🧠 **24 ML Models** — 3 variants × 8 ensemble models for robust CKD prediction
- 🔄 **3 Feature Selection Strategies** — All Features (24), RFE (12), Boruta (selected)
- ⚖️ **SMOTE Balancing** — Borderline-SMOTE with regular SMOTE fallback for imbalanced data
- 🌐 **REST API** — FastAPI-based prediction and training endpoints
- 📊 **Interactive Dashboard** — React + TypeScript frontend with real-time predictions
- 📈 **Rich Visualizations** — Confusion matrices, ROC curves, comparison charts
- 🔍 **MICE Imputation** — Advanced missing data handling
- 🏥 **Clinical Ready** — Based on Rahman et al., 2024 CKD research

---

## 🚀 Installation

### Prerequisites

- **Python** >= 3.10
- **Node.js** >= 18
- **npm** >= 9

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/H0NEYP0T-466/CKD-Ensemble-Classifier.git
cd CKD-Ensemble-Classifier

# Create virtual environment
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run ML pipeline (trains all 24 models)
python -m backend.ml_core.pipeline

# Start API server
python run.py
# Server runs at http://localhost:8007
```

### Frontend Setup

```bash
# From project root
npm install

# Start development server
npm run dev
# App runs at http://localhost:5173

# Build for production
npm run build
```

### Environment Variables

**Backend** (`.env`):
```env
PORT=8007
PYTHONPATH=backend
```

**Frontend** (`.env`):
```env
VITE_API_URL=http://localhost:8007
```

---

## ⚡ Usage

### 🔮 Making Predictions

**Via API:**
```bash
curl -X POST http://localhost:8007/predict/all_features \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "blood_pressure": 80,
    "specific_gravity": 1.02,
    "albumin": 0,
    "sugar": 0,
    "red_blood_cells": "normal",
    "pus_cell": "normal",
    "pus_cell_clumps": "notpresent",
    "bacteria": "notpresent",
    "blood_glucose_random": 90,
    "blood_urea": 20,
    "serum_creatinine": 0.8,
    "sodium": 140,
    "potassium": 4.2,
    "hemoglobin": 14,
    "packed_cell_volume": 42,
    "white_blood_cell_count": 7500,
    "red_blood_cell_count": 5.2,
    "hypertension": "no",
    "diabetes_mellitus": "no",
    "coronary_artery_disease": "no",
    "appetite": "good",
    "pedal_edema": "no",
    "anemia": "no"
  }'
```

**Via Web UI:**
Navigate to [ckd-ensemble-classifier.vercel.app](https://ckd-ensemble-classifier.vercel.app/) → Enter patient data → Get instant CKD prediction with risk indicators.

### 🏋️ Training Models

```bash
# Trigger full training pipeline via API
curl -X POST http://localhost:8007/train
```

### 📊 Viewing Metrics

```bash
# Get evaluation metrics for all variants
curl http://localhost:8007/metrics
```

---

## 🛠 Tech Stack

### Languages
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

### Frameworks & Libraries
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-00B4D8?style=for-the-badge&logo=lightgbm&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### DevOps / CI / Tools
![npm](https://img.shields.io/badge/npm-CB3837?style=for-the-badge&logo=npm&logoColor=white)
![pip](https://img.shields.io/badge/pip-3775A9?style=for-the-badge&logo=pypi&logoColor=white)
![ESLint](https://img.shields.io/badge/ESLint-4B32C3?style=for-the-badge&logo=eslint&logoColor=white)
![Prettier](https://img.shields.io/badge/Prettier-F7B93E?style=for-the-badge&logo=prettier&logoColor=black)

### Cloud / Hosting
![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

---

## 📦 Dependencies & Packages

### Runtime Dependencies

<details>
<summary><b>Python (backend/requirements.txt)</b></summary>

| Package | Version | Description |
|---------|---------|-------------|
| ![NumPy](https://img.shields.io/pypi/v/numpy?style=for-the-badge&label=numpy) | ≥1.26.0 | Numerical computing |
| ![Pandas](https://img.shields.io/pypi/v/pandas?style=for-the-badge&label=pandas) | ≥2.2.0 | Data manipulation |
| ![scikit-learn](https://img.shields.io/pypi/v/scikit-learn?style=for-the-badge&label=scikit-learn) | ≥1.4.0 | Machine learning core |
| ![SciPy](https://img.shields.io/pypi/v/scipy?style=for-the-badge&label=scipy) | ≥1.13.0 | Scientific computing |
| ![boruta](https://img.shields.io/pypi/v/boruta?style=for-the-badge&label=boruta) | ≥0.3 | Boruta feature selection |
| ![imbalanced-learn](https://img.shields.io/pypi/v/imbalanced-learn?style=for-the-badge&label=imbalanced-learn) | ≥0.12.0 | SMOTE oversampling |
| ![XGBoost](https://img.shields.io/pypi/v/xgboost?style=for-the-badge&label=xgboost) | ≥2.0.0 | Gradient boosting |
| ![LightGBM](https://img.shields.io/pypi/v/lightgbm?style=for-the-badge&label=lightgbm) | ≥4.3.0 | Light GBM boosting |
| ![Matplotlib](https://img.shields.io/pypi/v/matplotlib?style=for-the-badge&label=matplotlib) | ≥3.8.0 | Plotting & visualization |
| ![Seaborn](https://img.shields.io/pypi/v/seaborn?style=for-the-badge&label=seaborn) | ≥0.13.0 | Statistical visualization |
| ![FastAPI](https://img.shields.io/pypi/v/fastapi?style=for-the-badge&label=fastapi) | ≥0.111.0 | Web framework |
| ![Uvicorn](https://img.shields.io/pypi/v/uvicorn?style=for-the-badge&label=uvicorn) | ≥0.29.0 | ASGI server |
| ![joblib](https://img.shields.io/pypi/v/joblib?style=for-the-badge&label=joblib) | ≥1.4.0 | Model serialization |
| ![Pydantic](https://img.shields.io/pypi/v/pydantic?style=for-the-badge&label=pydantic) | ≥2.7.0 | Data validation |
| ![python-multipart](https://img.shields.io/pypi/v/python-multipart?style=for-the-badge&label=python-multipart) | ≥0.0.9 | Form parsing |

</details>

<details>
<summary><b>Node.js (package.json)</b></summary>

| Package | Version | Description |
|---------|---------|-------------|
| ![React](https://img.shields.io/npm/v/react?style=for-the-badge&label=react) | ^19.2.4 | UI framework |
| ![React DOM](https://img.shields.io/npm/v/react-dom?style=for-the-badge&label=react-dom) | ^19.2.4 | DOM rendering |
| ![React Router](https://img.shields.io/npm/v/react-router-dom?style=for-the-badge&label=react-router-dom) | ^7.14.1 | Client-side routing |
| ![TanStack Query](https://img.shields.io/npm/v/@tanstack/react-query?style=for-the-badge&label=%40tanstack%2Freact-query) | ^5.99.0 | Server state management |
| ![Axios](https://img.shields.io/npm/v/axios?style=for-the-badge&label=axios) | ^1.6.8 | HTTP client |

</details>

### Dev / Build / Test Dependencies

<details>
<summary><b>Node.js Dev Dependencies</b></summary>

| Package | Version | Description |
|---------|---------|-------------|
| ![TypeScript](https://img.shields.io/npm/v/typescript?style=for-the-badge&label=typescript) | ~6.0.2 | Type-safe JavaScript |
| ![Vite](https://img.shields.io/npm/v/vite?style=for-the-badge&label=vite) | ^8.0.4 | Build tool |
| ![ESLint](https://img.shields.io/npm/v/eslint?style=for-the-badge&label=eslint) | ^9.39.9 | Linter |
| ![vite-plugin-react](https://img.shields.io/npm/v/@vitejs/plugin-react?style=for-the-badge&label=%40vitejs%2Fplugin-react) | ^6.0.1 | React fast refresh |
| ![typescript-eslint](https://img.shields.io/npm/v/typescript-eslint?style=for-the-badge&label=typescript-eslint) | ^8.58.0 | TS ESLint rules |
| ![eslint-plugin-react-hooks](https://img.shields.io/npm/v/eslint-plugin-react-hooks?style=for-the-badge&label=eslint-plugin-react-hooks) | ^7.0.1 | React hooks linting |
| ![eslint-plugin-react-refresh](https://img.shields.io/npm/v/eslint-plugin-react-refresh?style=for-the-badge&label=eslint-plugin-react-refresh) | ^0.5.2 | React refresh linting |
| ![globals](https://img.shields.io/npm/v/globals?style=for-the-badge&label=globals) | ^17.4.0 | Global variable definitions |
| ![Node types](https://img.shields.io/npm/v/@types/node?style=for-the-badge&label=%40types%2Fnode) | ^24.12.2 | Node.js type definitions |
| ![React types](https://img.shields.io/npm/v/@types/react?style=for-the-badge&label=%40types%2Freact) | ^19.2.14 | React type definitions |
| ![React DOM types](https://img.shields.io/npm/v/@types/react-dom?style=for-the-badge&label=%40types%2Freact-dom) | ^19.2.3 | React DOM type definitions |

</details>

---

## 📂 Folder Structure

```
CKD-Ensemble-Classifier/
├── 📁 backend/                     # FastAPI ML Backend
│   ├── 📁 api/                     # API layer
│   │   ├── main.py                 # FastAPI application
│   │   └── routers.py              # API route handlers
│   ├── 📁 ml_core/                 # ML pipeline core
│   │   ├── pipeline.py             # Main orchestrator (trains all 24 models)
│   │   ├── preprocess.py           # Data preprocessing (MICE, scaling)
│   │   ├── feature_selection.py    # RFE & Boruta feature selection
│   │   ├── train.py                # Model training (8 ensemble models)
│   │   └── evaluate.py             # Evaluation & visualization
│   ├── 📁 schemas/                 # Pydantic data models
│   │   └── models.py               # Request/response schemas
│   ├── 📁 Dataset/                 # CKD dataset files
│   ├── 📁 models/                  # Trained model serialization
│   ├── 📁 plots/                   # Generated visualizations
│   │   ├── 📁 all_features/        # Charts for all-features variant
│   │   ├── 📁 rfe/                 # Charts for RFE variant
│   │   └── 📁 boruta/              # Charts for Boruta variant
│   ├── requirements.txt            # Python dependencies
│   └── run.py                      # Server entry point
│
├── 📁 src/                         # React Frontend
│   ├── 📁 components/              # Reusable UI components
│   │   ├── FormField.tsx           # Input form field
│   │   ├── LoadingSpinner.tsx      # Loading indicator
│   │   ├── MetricsTable.tsx        # Metrics display table
│   │   ├── Navbar.tsx              # Navigation bar
│   │   └── RiskIndicator.tsx       # CKD risk visualization
│   ├── 📁 pages/                   # Page components
│   │   ├── PredictionPage.tsx      # Prediction form page
│   │   └── AnalyticsPage.tsx       # Model analytics page
│   ├── 📁 services/                # API client
│   │   └── api.ts                  # Axios API service
│   ├── 📁 types/                   # TypeScript types
│   │   └── index.ts                # Type definitions
│   ├── App.tsx                     # Root application component
│   ├── main.tsx                    # Application entry point
│   └── assets/                     # Static assets
│
├── 📁 .github/                     # GitHub configuration
│   ├── 📁 ISSUE_TEMPLATE/          # Issue form templates
│   │   ├── bug_report.yml          # Bug report form
│   │   ├── feature_request.yml     # Feature request form
│   │   └── config.yml              # Template config
│   └── pull_request_template.md    # PR template
│
├── CLAUDE.md                       # AI assistant instructions
├── package.json                    # Node.js dependencies
├── vite.config.ts                  # Vite configuration
├── eslint.config.js                # ESLint configuration
├── tsconfig.json                   # TypeScript configuration
├── LICENSE                         # MIT License
├── CONTRIBUTING.md                 # Contribution guidelines
├── SECURITY.md                     # Security policy
├── CODE_OF_CONDUCT.md              # Community standards
└── README.md                       # This file
```

---

## 📊 Model Performance & Charts

The ML pipeline generates comprehensive visualizations for each of the 3 variants (all_features, rfe, boruta):

### 📈 Comparison Charts

#### Accuracy Comparison
![Accuracy Comparison](backend/plots/all_features/comparison_accuracy.png)

#### AUC-ROC Comparison
![AUC-ROC Comparison](backend/plots/all_features/comparison_auc_roc.png)

#### F1 Score Comparison
![F1 Score Comparison](backend/plots/all_features/comparison_f1_score.png)

#### Precision Comparison
![Precision Comparison](backend/plots/all_features/comparison_precision.png)

#### Recall Comparison
![Recall Comparison](backend/plots/all_features/comparison_recall.png)

#### All Metrics Comparison
![All Metrics Comparison](backend/plots/all_features/comparison_all_metrics.png)

### 🔄 ROC Curves

![ROC Curves](backend/plots/all_features/roc_curves.png)

### 📉 Confusion Matrices

| Random Forest | XGBoost | LightGBM | GBDT |
|---|---|---|---|
| ![RF](backend/plots/all_features/cm_Random_Forest.png) | ![XGB](backend/plots/all_features/cm_XGBoost.png) | ![LGBM](backend/plots/all_features/cm_LightGBM.png) | ![GBDT](backend/plots/all_features/cm_GBDT.png) |

| AdaBoost | Bagging | Stacking | Voting Soft |
|---|---|---|---|
| ![Ada](backend/plots/all_features/cm_AdaBoost.png) | ![Bag](backend/plots/all_features/cm_Bagging.png) | ![Stack](backend/plots/all_features/cm_Stacking.png) | ![Vote](backend/plots/all_features/cm_Voting_Soft.png) |

### 📊 Data Distribution

![KDE Distributions](backend/plots/kde_distributions.png)

> **Note:** Similar charts are generated for all three variants (`all_features`, `rfe`, `boruta`) and stored in their respective directories under `backend/plots/`.

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/train` | Trigger full ML pipeline training |
| `POST` | `/predict/{variant}` | Get CKD prediction (`all_features`, `rfe`, or `boruta`) |
| `GET` | `/metrics` | Get evaluation metrics for all variants |

### Prediction Request Body

```json
{
  "age": 45,
  "blood_pressure": 80,
  "specific_gravity": 1.02,
  "albumin": 0,
  "sugar": 0,
  "red_blood_cells": "normal",
  "pus_cell": "normal",
  "pus_cell_clumps": "notpresent",
  "bacteria": "notpresent",
  "blood_glucose_random": 90,
  "blood_urea": 20,
  "serum_creatinine": 0.8,
  "sodium": 140,
  "potassium": 4.2,
  "hemoglobin": 14,
  "packed_cell_volume": 42,
  "white_blood_cell_count": 7500,
  "red_blood_cell_count": 5.2,
  "hypertension": "no",
  "diabetes_mellitus": "no",
  "coronary_artery_disease": "no",
  "appetite": "good",
  "pedal_edema": "no",
  "anemia": "no"
}
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🛡 Security

Please review our [Security Policy](SECURITY.md) for information on reporting vulnerabilities and our responsible disclosure process.

---

## 📏 Code of Conduct

We are committed to providing a welcoming and inclusive experience. Please read our [Code of Conduct](CODE_OF_CONDUCT.md).

---

<p align="center">Made with ❤ by <a href="https://github.com/H0NEYP0T-466">H0NEYP0T-466</a></p>
