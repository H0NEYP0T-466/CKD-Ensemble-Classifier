# 🤝 Contributing to CKD Ensemble Classifier

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## 🚀 Getting Started

### 1. Fork & Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/<your-username>/CKD-Ensemble-Classifier.git
cd CKD-Ensemble-Classifier
```

### 2. Set Up Development Environment

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend (from project root)
npm install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 📝 Code Style & Linting

### Python
- **Formatting:** Black
- **Linting:** Ruff
- **Max line length:** 88 characters

```bash
cd backend
black .
ruff check .
```

### TypeScript / React
- **Formatting:** Prettier
- **Linting:** ESLint
- **Strict TypeScript:** Enabled

```bash
npm run lint
```

## 🐛 Bug Reports

If you find a bug, please [open an issue](https://github.com/H0NEYP0T-466/CKD-Ensemble-Classifier/issues/new?template=bug_report.yml) with:

1. **Clear description** of the problem
2. **Steps to reproduce** the issue
3. **Expected behavior** vs. **actual behavior**
4. **Environment details** (OS, Python/Node versions)
5. **Logs or screenshots** if applicable

## 💡 Feature Requests

To suggest a new feature, [open a feature request](https://github.com/H0NEYP0T-466/CKD-Ensemble-Classifier/issues/new?template=feature_request.yml) with:

1. **Problem statement** — What problem does this solve?
2. **Proposed solution** — Your idea for the feature
3. **Alternatives considered** — Other approaches you've thought about
4. **Scope** — What's in and out of scope

## 🧪 Testing

### Python Tests
```bash
cd backend
pytest --cov=ml_core --cov-report=term-missing
```

### TypeScript / React Tests
```bash
npm test
```

> **Coverage requirement:** 80%+ test coverage for all new code.

## 📖 Documentation

- Update the `README.md` for user-facing changes
- Update `CLAUDE.md` for architectural or workflow changes
- Add docstrings to all public Python functions
- Update TypeScript types for API changes

## 🔄 Pull Request Process

1. **Ensure all tests pass** before submitting
2. **Update documentation** to reflect your changes
3. **Follow conventional commits** format:
   - `feat:` — New feature
   - `fix:` — Bug fix
   - `refactor:` — Code refactoring
   - `docs:` — Documentation update
   - `test:` — Test additions/changes
   - `chore:` — Maintenance tasks
4. **Fill out the PR template** completely
5. **Link related issues** (e.g., "Fixes #123")
6. **Request review** from maintainers

## 🏗️ Project Structure Notes

- **Backend ML code** → `backend/ml_core/`
- **API endpoints** → `backend/api/`
- **Frontend components** → `src/components/`
- **Frontend pages** → `src/pages/`
- **TypeScript types** → `src/types/`

## 📜 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🛡 Security

If you discover a security vulnerability, please see our [Security Policy](SECURITY.md) for responsible disclosure.

---

<p align="center">Thank you for contributing! 🎉</p>
