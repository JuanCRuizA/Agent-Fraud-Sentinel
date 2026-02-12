# BAFS - Deployment Guide

## Purpose

Document deployment options, configurations, and operational procedures for the BAFS fraud detection dashboard.

---

## Current Status

| Item | Detail |
|------|--------|
| **Live URL** | [bankingantifraudsystem.streamlit.app](https://bankingantifraudsystem.streamlit.app) |
| **Platform** | Streamlit Cloud (free tier) |
| **Main file** | `notebooks/dashboard/dashboard_app.py` |
| **Deployed from** | `main` branch, auto-redeploys on push |
| **Date** | 2026-02-10 |

---

## Deployment Options Overview

This guide covers three deployment approaches for the BAFS fraud detection dashboard:

1. **Streamlit Cloud** -- Recommended for portfolio demos (current deployment)
2. **Local Docker Container** -- For development and testing
3. **Enterprise Cloud Deployment** -- AWS ECS / GCP Cloud Run (production considerations)

---

## 1. Streamlit Cloud Deployment

### Why Streamlit Cloud?

- Zero infrastructure management
- Free tier available
- Automatic deployments from GitHub
- Built-in HTTPS
- Ideal for portfolio demonstrations

### Prerequisites

- GitHub account with repository access
- Streamlit Cloud account (free tier: [share.streamlit.io](https://share.streamlit.io))
- All required files committed to GitHub

### Critical Files Checklist

**Must be committed to GitHub:**

| File | Size | Purpose |
|------|------|---------|
| `models/xgboost_final.pkl` | < 1 MB | Trained XGBoost model |
| `models/scaler.pkl` | < 1 MB | Feature scaler |
| `models/threshold_config.pkl` | < 1 MB | Threshold configuration |
| `notebooks/dashboard/test_dashboard.csv` | 4.2 MB | Slim test data (8 columns) |
| `notebooks/dashboard/dashboard_app.py` | ~20 KB | Streamlit application |
| `notebooks/dashboard/requirements.txt` | < 1 KB | Python dependencies |
| `figures/shap/*.png` | ~3 MB total | SHAP explainability figures |

**Important:**

- `.gitignore` must allow model `.pkl` files to be tracked (line `models/*.pkl` commented out)
- Large data files (>100 MB) must be slimmed down or excluded
- See [ISSUE-010](issues-solutions.md#issue-010-streamlit-cloud-filenotfounderror-for-model-artifacts) and [ISSUE-011](issues-solutions.md#issue-011-test-csv-exceeds-github-100-mb-file-size-limit) in `issues-solutions.md` for related troubleshooting

### Deployment Steps

**Step 1: Prepare Repository**

```bash
# Verify all required files are committed
git status
git add models/ notebooks/dashboard/ figures/shap/
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

**Step 2: Connect to Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select repository: `JuanCRuizA/Agent-Fraud-Sentinel`
5. Set main file path: `notebooks/dashboard/dashboard_app.py`
6. Set Python version: 3.10

**Step 3: Configure Environment**

- Streamlit Cloud automatically installs packages from `notebooks/dashboard/requirements.txt`
- No additional environment variables needed for BAFS

**Step 4: Deploy**

1. Click "Deploy"
2. Wait 2-5 minutes for initial build
3. App will be live at [bankingantifraudsystem.streamlit.app](https://bankingantifraudsystem.streamlit.app)

### Monitoring and Updates

**View Logs:**
Click "Manage app" > "Logs" to see real-time deployment logs. Useful for debugging import errors or missing files.

**Automatic Updates:**
Every `git push` to `main` triggers automatic redeployment. Changes typically reflect within 2-3 minutes.

**Resource Limits (Free Tier):**

| Resource | Limit |
|----------|-------|
| RAM | 1 GB |
| CPU | Shared |
| Apps per account | 1 (can delete and redeploy) |

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: models/xgboost_final.pkl` | `.gitignore` excludes model files | Comment out `models/*.pkl` in `.gitignore`. See [ISSUE-010](issues-solutions.md) |
| `MemoryError` during model loading | Full `test.csv` (145 MB) exceeds free tier | Use `test_dashboard.csv` (4.2 MB slim version). See [ISSUE-011](issues-solutions.md) |
| Import errors for packages | Missing dependencies | Verify all packages in `notebooks/dashboard/requirements.txt` |

---

## 2. Local Docker Deployment

### Why Docker?

- Consistent environment across machines
- Easy sharing with recruiters (send Dockerfile)
- Simulates production deployment
- Demonstrates DevOps knowledge

### Prerequisites

- Docker Desktop installed ([docker.com](https://www.docker.com))
- Docker daemon running

### Dockerfile

Create `Dockerfile` in the project root:

```dockerfile
# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY notebooks/dashboard/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "notebooks/dashboard/dashboard_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build Docker image
cd /path/to/agent-fraud-sentinel
docker build -t bafs-fraud-detection:v1.0 .

# Run container
docker run -p 8501:8501 bafs-fraud-detection:v1.0

# Access dashboard at http://localhost:8501

# Stop container
docker ps                      # find container ID
docker stop <container_id>
```

### Docker Best Practices

**1. Use `.dockerignore`**

Create `.dockerignore` to exclude unnecessary files:

```
.git
.venv
__pycache__
*.pyc
.DS_Store
data/raw/*
.vscode
.claude
```

**2. Multi-stage builds (for smaller image size)**

```dockerfile
# Builder stage
FROM python:3.10 AS builder
WORKDIR /app
COPY notebooks/dashboard/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["streamlit", "run", "notebooks/dashboard/dashboard_app.py"]
```

**3. Volume mounting for development**

```bash
# Mount local directory to test changes without rebuilding
docker run -p 8501:8501 -v $(pwd):/app bafs-fraud-detection:v1.0
```

---

## 3. Enterprise Cloud Deployment (Production Considerations)

> **Note:** This section outlines how BAFS could scale to a production banking environment. It is included to demonstrate architectural awareness and is not implemented in the current portfolio version.

### AWS ECS (Elastic Container Service)

**High-Level Architecture:**

```
User --> ALB (Load Balancer) --> ECS Task (Docker Container) --> RDS (Optional, for logging)
```

**Steps (Summary):**

1. Push Docker image to Amazon ECR
2. Create ECS Task Definition with container specs
3. Configure Application Load Balancer
4. Set up Auto Scaling based on CPU/memory
5. Configure CloudWatch for monitoring

**Estimated Monthly Cost:** $30-50 for low-traffic portfolio demo

### GCP Cloud Run

**Why Cloud Run?**

- Serverless (no server management)
- Pay-per-request pricing
- Automatic scaling to zero when idle

**Deployment Command:**

```bash
gcloud run deploy bafs-dashboard \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**Estimated Monthly Cost:** $5-15 for portfolio demo usage

---

## Security Considerations

### For Production Deployments

| Area | Recommendation |
|------|----------------|
| Authentication | Integrate with OAuth 2.0 or SAML |
| Secrets | Store in AWS Secrets Manager / GCP Secret Manager |
| Rate Limiting | Implement API rate limits to prevent abuse |
| Logging | Forward logs to centralized service (Datadog, Splunk) |
| Model Versioning | Store model artifacts in S3/GCS with versioning enabled |

### For Portfolio Demos

- Public access is acceptable (no sensitive data in the dataset)
- Basic authentication optional via Streamlit's native auth

### General Best Practices

**Never commit to GitHub:**
- API keys, database credentials, private encryption keys, internal IP addresses

**Use environment variables for sensitive configuration:**

```python
import os
SECRET_KEY = os.getenv("SECRET_KEY", "default-for-local-dev")
```

**Model integrity:**
- Hash model files (SHA256) to detect tampering
- Log model version used for each prediction

---

## Testing Before Deployment

### Unit Tests

Create `tests/test_predictions.py`:

```python
import pytest
import joblib
import pandas as pd

def test_model_loads():
    """Verify model file exists and loads correctly."""
    model = joblib.load('models/xgboost_final.pkl')
    assert model is not None

def test_scaler_loads():
    """Verify scaler file exists and loads correctly."""
    scaler = joblib.load('models/scaler.pkl')
    assert scaler is not None

def test_prediction_shape():
    """Verify model returns correct output shape."""
    model = joblib.load('models/xgboost_final.pkl')

    # Use the actual 7 features from training
    sample = pd.DataFrame([[3, 8, 1.5, 0, 14, 1, 75.0]],
                          columns=['txn_count_1hr', 'txn_count_24hr',
                                   'amount_deviation', 'is_first_transaction',
                                   'hour_of_day', 'is_weekend',
                                   'TransactionAmt'])

    pred = model.predict_proba(sample)
    assert pred.shape == (1, 2)
```

Run tests:

```bash
pytest tests/test_predictions.py
```

### Integration Tests (Manual Checklist)

- [ ] Dashboard loads successfully in browser
- [ ] Sidebar navigation switches between all 4 pages
- [ ] Threshold slider updates predictions dynamically
- [ ] SHAP figures render without errors on Model Performance page
- [ ] Case Study Explorer displays all 6 case studies
- [ ] Dashboard loads in under 3 seconds on second visit (cached)

---

## Post-Deployment Monitoring

### For Streamlit Cloud (Current)

| Metric | How to Check |
|--------|-------------|
| Unique visitors | Streamlit Cloud analytics dashboard |
| Average session duration | Streamlit Cloud analytics dashboard |
| Deployment status | "Manage app" > status indicator |

### For Production (Future Considerations)

| Metric | Target |
|--------|--------|
| Prediction Latency | < 200ms (p95) |
| Error Rate | < 0.1% |
| Model Drift | Monthly PR-AUC comparison |
| System Uptime | 99.5%+ |

---

## Model Retraining Workflow (Future Enhancement)

### When to Retrain

- Monthly (scheduled)
- When PR-AUC drops > 5% on validation set
- After major fraud pattern changes detected

### Steps

1. Collect new transaction data (last 30 days)
2. Retrain XGBoost with same hyperparameters
3. Compare PR-AUC on holdout set
4. If improvement > 2%, deploy new model
5. Document in [technical-decisions.md](technical-decisions.md)

### Deployment Strategy

- **Blue-Green deployment:** Keep old model live until new model is validated
- **Canary deployment:** Route 10% traffic to new model initially

---

## Deployment Checklist

### Before Deploying

- [ ] All tests passing
- [ ] Model files committed to GitHub (< 100 MB total)
- [ ] `notebooks/dashboard/requirements.txt` up to date
- [ ] `.gitignore` configured correctly (model `.pkl` files allowed)
- [ ] README.md includes live demo link
- [ ] This guide references correct file paths

### After Deploying

- [ ] Dashboard loads successfully at [bankingantifraudsystem.streamlit.app](https://bankingantifraudsystem.streamlit.app)
- [ ] Test prediction with sample transaction
- [ ] SHAP figures render on Model Performance page
- [ ] Share demo link with 2-3 peers for feedback
- [ ] Add link to resume and LinkedIn

---

## Related Documentation

- [issues-solutions.md](issues-solutions.md) -- ISSUE-009 through ISSUE-011 cover deployment-related problems
- [technical-decisions.md](technical-decisions.md) -- DECISION-013 covers slim test data strategy
- [notebooks-progress.md](notebooks-progress.md) -- Phase 5 entry documents dashboard architecture
