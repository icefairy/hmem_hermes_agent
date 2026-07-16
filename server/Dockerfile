# ============================================================
# Stage 1: Build WebUI (Vue3 SPA)
# ============================================================
FROM node:20-slim AS webui-builder

WORKDIR /app/webui

# 依赖安装
COPY server/webui/package.json server/webui/pnpm-lock.yaml* ./
RUN npm install -g pnpm && pnpm install --frozen-lockfile 2>/dev/null || pnpm install

# 构建
COPY server/webui/ .
RUN pnpm build

# ============================================================
# Stage 2: Runtime (Python)
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制 WebUI 构建产物
COPY --from=webui-builder /app/webui/dist /app/webui/dist

# 复制 Server 代码
COPY server/ /app/

# 数据目录
VOLUME /data/hmem

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]