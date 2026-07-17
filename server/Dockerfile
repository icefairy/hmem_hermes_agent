FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

# 预安装 Python 依赖（Dockerfile 层缓存）
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

VOLUME /data/hmem

# --- 构建产物的最小生产镜像 ---
FROM base AS prod

COPY server/ /app/
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]

# --- 开发镜像：代码通过 volume 挂载 ---
FROM base AS dev

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--reload"]