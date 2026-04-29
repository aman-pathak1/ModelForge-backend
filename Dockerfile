FROM python:3.11-slim

WORKDIR /app

<<<<<<< HEAD
# ── System dependencies for LightGBM (libgomp1) and XGBoost ──────
=======
>>>>>>> 313846fba3a1ca89418213fd608070b76aece72e
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
