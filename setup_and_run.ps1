# Setup virtual environment and run the RAG agent (PowerShell)
# Run from project root: .\setup_and_run.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 1. Create venv if not present
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

# 2. Activate and install requirements
Write-Host "Activating venv and installing requirements (this may take a few minutes)..."
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Build index if chroma_db is missing or empty
if (-not (Test-Path "chroma_db") -or -not (Get-ChildItem "chroma_db" -ErrorAction SilentlyContinue)) {
    Write-Host "Building RAG index (FEVER + RAGTruth)..."
    python build_index.py
} else {
    Write-Host "Using existing chroma_db index. To rebuild, delete chroma_db and run: python build_index.py"
}

# 4. Run agent with a sample query
Write-Host "Running agent with sample query..."
python run_agent.py "What is fact verification? Give one example from the knowledge base."

Write-Host "`nDone. To ask your own question: python run_agent.py `"Your question`""
