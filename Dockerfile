# -*- coding: utf-8 -*-
# 应用镜像：只装 Python 依赖与代码，不装 MySQL/Milvus/Redis（那些由 compose 提供）。
#
# 为什么用 python:3.12-slim 而不是 alpine：
#   - alpine 用 musl libc，torch / onnxruntime / grpcio 这些有 C 扩展的包没有预编译 wheel，
#     装的时候要从源码编译，镜像能构建成功但耗时 40 分钟以上，换台机器还可能编译失败。
#   - slim 是 glibc，所有包都有官方 wheel，装得快也稳。
#   代价是基础镜像大 60MB，这个项目里完全划算。
FROM python:3.12-slim

# PYTHONDONTWRITEBYTECODE：容器里不需要 .pyc，省空间
# PYTHONUNBUFFERED：日志直接输出到 stdout，不缓冲，docker logs 才能实时看到
# PIP_NO_CACHE_DIR：不在镜像里留 pip 下载缓存，镜像能小几百 MB
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=utf-8 \
    TZ=Asia/Shanghai

# libgomp1：torch 在 CPU 上跑要这个 OpenMP 运行库，slim 镜像默认没有
# curl：下面 HEALTHCHECK 要用
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先只拷 requirements.txt 再装依赖：
# Docker 分层缓存，只要 requirements.txt 没变，改代码重建时不会重装依赖。
# 如果一开始就 COPY . . ，任何一次代码改动都会让后面所有层失效，装依赖要重来一遍。
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# torch 必须从 CPU 专用源装。
# 直接 pip install torch 在 x86_64 Linux 上会装成 CUDA 版，连带拉进
# nvidia-cudnn / nvidia-cublas / triton / cuda-toolkit 等 18 个包约 5GB，
# 而本项目是纯 CPU 推理（embedder.py 里 device 判断只会得出 cpu），这 5GB 全是死重。
# 用 +cpu 索引后体积从 6.9GB 降到约 1.6GB，构建时间也从 8 分钟降到 3 分钟以内。
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.1.0" \
    && pip install --no-cache-dir \
        "sentence-transformers>=2.2.0" \
        "milvus-model>=0.2.0" \
        "FlagEmbedding>=1.2.10"

# 再拷代码。.dockerignore 已经排除了 models/ logs/ .venv 等目录
COPY . .

# 以非 root 用户运行。容器里跑的是网络服务，万一被攻破，root 权限能把宿主机挂载
# 进来的目录一起改掉，非 root 只影响容器内部。
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/logs /app/data/processed \
    && chown -R appuser:appuser /app
USER appuser

# 服务端口。uvicorn 用 0.0.0.0（容器内必须监听所有网卡，否则宿主机连不进来），
# 跟本地跑 api.py 时的 127.0.0.1 不一样——这是容器化最常踩的坑之一。
EXPOSE 8000

# 健康检查：compose 的 depends_on.condition: service_healthy 靠这个判断
# 启动顺序。这里查 /api/overview，它同时验证了数据库连通性，比只查端口更准。
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/overview || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
