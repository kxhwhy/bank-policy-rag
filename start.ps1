# 启动银行制度问答服务
#
# 用法：先把密钥写进 .env（或在本进程里 $env:POLICY_LLM_API_KEY = "..."），再执行  .\start.ps1
#       .env 已在 .gitignore 里，不会进仓库。

$ErrorActionPreference = "Stop"

# ---------- 读取 .env ----------
# 密钥不写死在本文件：这个文件是要提交到仓库的，写死等于把 key 公开。
# .env 由使用者自己创建（config.ini 里同名项留空也一样能跑，只是模型不可用）。
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile -Encoding UTF8 | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $idx = $line.IndexOf("=")
            $name = $line.Substring(0, $idx).Trim()
            $value = $line.Substring($idx + 1).Trim().Trim('"').Trim("'")
            if ($name -and -not (Test-Path "env:$name")) {
                Set-Item -Path "env:$name" -Value $value
            }
        }
    }
}

# ---------- 本地默认值（不含密钥） ----------
if (-not $env:POLICY_MYSQL_PASSWORD) { $env:POLICY_MYSQL_PASSWORD = "root" }
if (-not $env:POLICY_REDIS_PASSWORD) { $env:POLICY_REDIS_PASSWORD = "1234" }

if (-not $env:POLICY_LLM_BASE_URL) { $env:POLICY_LLM_BASE_URL = "https://api.deepseek.com/v1" }
if (-not $env:POLICY_LLM_MODEL)    { $env:POLICY_LLM_MODEL    = "deepseek-chat" }

# 中文日志在 Windows 控制台会因编码报 UnicodeEncodeError，必须设成 utf-8
$env:PYTHONIOENCODING = "utf-8"

Set-Location $PSScriptRoot

Write-Host ""
Write-Host "  存储      MySQL 127.0.0.1:3306  ·  Redis 127.0.0.1:6379/db3  ·  Milvus 127.0.0.1:19530" -ForegroundColor DarkGray
Write-Host "  模型      $env:POLICY_LLM_MODEL  ($env:POLICY_LLM_BASE_URL)" -ForegroundColor DarkGray
if (-not $env:POLICY_LLM_API_KEY) {
    Write-Host "  警告      未检测到 POLICY_LLM_API_KEY，将降级为条款原文直读" -ForegroundColor Yellow
}
Write-Host "  访问      http://127.0.0.1:8000/" -ForegroundColor Cyan
Write-Host "  提示      首次提问要加载 BGE-M3 与 reranker，约 30 秒，之后毫秒级" -ForegroundColor DarkGray
Write-Host ""

python api.py
