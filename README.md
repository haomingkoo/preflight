# EDA + Classification Dash

A Dash-based EDA and classification workbench for quick, guided exploration and baseline modeling.

## What this app does

- Upload CSV or Parquet
- Basic data health summary (missingness, uniques, duplicates)
- Missingness matrix + top missing columns
- Feature typing suggestions with overrides
- EDA charts per feature (numeric histograms, categorical counts/rates)
- Correlation heatmap + top correlated-pair scatters
- Baseline models with CV metrics + confusion matrix + ROC/PR where applicable

---

# 1) Run locally (recommended for large files)

## Prerequisites

- Python 3.12.x recommended
- pip available
- macOS/Linux/WSL supported

## Setup

From the `spotlight_classification/` folder:

```bash
python -V
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```bash
python app.py
```

## open in your browser
```bash
http://localhost:8050
```

---

# 2) Run with Docker (reproducible, portable)

## Prerequisites
- Docker installed
- Access to Docker daemon (Docker Desktop, Colima, Rancher Desktop, or Otherside cpubox)

## Build the image
From the spotlight_classification/ folder (where the Dockerfile is):
```bash
docker build -t spotlight_classification:latest .
```

## Run the container
Choose a free host port (example uses 8050):
```bash
docker run -d --rm \
  --name spotlight_classification \
  -p 8050:8050 \
  spotlight_classification:latest
```

## Notes on ports and shared environments (Otherside)
- The app listens inside the container on port 8050
- You may change the host port if 8050 is already in use:

In Otherside:
- Docker runs on cpubox
- You must forward ports to devbox using socat

Example (White space sensitive):
```bash
socat TCP-LISTEN:8050,fork TCP:cpubox:8050 &
```

## open in your browser
```bash
http://localhost:8050
```

### Accessing the app via Coder (Shared Ports)

This application can be accessed through Coder Shared Ports, which allow HTTP services running inside your workspace to be exposed to other users or the public.

What are Shared Ports?

Shared Ports are ports explicitly published via the Coder UI. Once shared, they can be accessed:
- By organization members
- By other authenticated Coder users
- Or publicly (depending on permissions)

Only ports that you explicitly share are accessible externally.


How to enable a Shared Port in Coder
1.	Open your Coder Workspace
2.	Click “Open ports” (top-right of the workspace UI)
3.	Scroll to Shared Ports
4.	Add a port:
    -	Port: 8050
    -	Protocol: HTTP
    -	Access: Authenticated (or Public if instructed)
5.	Confirm the port is listed under Shared Ports

Once enabled, Coder automatically generates a public URL

```bash
https://8050--main--haomingkoo--haoming-koo.coder.aiap21-aut0.aisingapore.net/
```