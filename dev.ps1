<#
.SYNOPSIS
  本地开发启动脚本
  覆写 Docker 特定的环境变量为本地值，启动 uvicorn 热重载
.DESCRIPTION
  用法: .\dev.ps1
  前置条件: PostgreSQL 和 Redis 已通过 docker-compose 运行
  自动从 .env 继承 API Key、JWT 等配置
#>

# ==================== 本地覆写（Docker 路径 → 本地路径） ====================
$env:POSTGRES_HOST   = "localhost"
$env:POSTGRES_PORT   = "5433"
$env:REDIS_HOST      = "localhost"
$env:REDIS_PASSWORD  = "redis123456"

$env:EMBEDDING_MODEL_PATH  = "D:\hf_models\BAAI\bge-m3"
$env:RERANKER_MODEL_PATH   = "D:\hf_models\BAAI\bge-reranker-v2-m3"
$env:EMBEDDING_ONNX_PATH   = "D:\hf_models\BAAI\bge-m3-onnx\bge-m3.onnx"
$env:RERANKER_ONNX_PATH    = "D:\hf_models\BAAI\bge-reranker-v2-m3-onnx\bge-reranker-v2-m3.onnx"

$env:CHROMA_PERSIST_DIR    = "D:\School\AI_rag\backend\vector_store\chroma"
$env:UPLOAD_DIR            = "D:\School\AI_rag\backend\uploads"

# ==================== CUDA 路径（让 onnxruntime-gpu 找到 PyTorch 自带的 cuDNN） ====================
$torchLib = "D:\miniconda3\envs\py312\Lib\site-packages\torch\lib"
if (Test-Path $torchLib) {
    $env:PATH = "$torchLib;$env:PATH"
    Write-Host "cuDNN:      $torchLib" -ForegroundColor Cyan
}

# ==================== 切换到 backend 目录（main.py 所在位置） ====================
Set-Location -LiteralPath "D:\School\AI_rag\backend"

# ==================== 启动后端 ====================
Write-Host "Starting backend with hot-reload..." -ForegroundColor Green
Write-Host "PostgreSQL: $env:POSTGRES_HOST`:$env:POSTGRES_PORT" -ForegroundColor Cyan
Write-Host "Redis:      $env:REDIS_HOST`:$env:REDIS_PORT" -ForegroundColor Cyan
Write-Host "ONNX:       $env:EMBEDDING_ONNX_PATH" -ForegroundColor Cyan
Write-Host "WorkDir:    $(Get-Location)" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan

uvicorn main:app --host 0.0.0.0 --port 8000

# ==================== 退出后返回项目根 ====================
Set-Location -LiteralPath "D:\School\AI_rag"
