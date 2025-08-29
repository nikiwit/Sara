# Configuration and System Information

Essential commands and utilities for system administration, model management, and performance monitoring in ML/LLM development environments.

## Table of Contents

- [Mac System Information](#mac-system-information)
  - [Hardware Specs](#hardware-specs)
  - [GPU and ML Acceleration](#gpu-and-ml-acceleration)
- [Model Management](#model-management)
  - [Ollama Models](#ollama-models)
  - [HuggingFace Models (Embedding/Reranker)](#huggingface-models-embeddingreranker)
  - [spaCy Models](#spacy-models)
  - [Current System Configuration](#current-system-configuration)
- [Environment and Configuration](#environment-and-configuration)
  - [Environment Files](#environment-files)
  - [Python Environment](#python-environment)
- [Performance Monitoring](#performance-monitoring)
  - [Resource Usage](#resource-usage)
  - [ML Workload Monitoring](#ml-workload-monitoring)
- [Development and Testing](#development-and-testing)
  - [Model Testing Commands](#model-testing-commands)
  - [Sara Application Commands](#sara-application-commands)
- [Useful ML/LLM Commands](#useful-mlllm-commands)
  - [Model Comparison and Selection](#model-comparison-and-selection)
  - [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Performance Optimization](#performance-optimization)

---

## Mac System Information

### Hardware Specs
```bash
# Full hardware profile
system_profiler SPHardwareDataType

# Memory information
system_profiler SPMemoryDataType

# Software version
system_profiler SPSoftwareDataType

# Condensed hardware info
system_profiler SPHardwareDataType SPMemoryDataType | grep -A 5 -E "(Model Name|Total Memory|Chip|System Version)"

# CPU information
sysctl -n machdep.cpu.brand_string
sysctl -n hw.physicalcpu hw.logicalcpu

# Memory usage
vm_stat | head -10

# Disk usage
df -h

# Temperature (if available)
sudo powermetrics --samplers smc -n 1 2>/dev/null | grep -i temp
```

### GPU and ML Acceleration
```bash
# Check Metal Performance Shaders availability
system_profiler SPDisplaysDataType | grep -A 5 "Metal"

# Python check for MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}' if hasattr(torch.backends, 'mps') else 'MPS not available')"

# Check CUDA availability (unlikely on Mac)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Model Management

### Ollama Models
```bash
# List all installed models
ollama list

# Check model details
ollama show MODEL_NAME

# Pull/download new model
ollama pull MODEL_NAME

# Remove model
ollama rm MODEL_NAME

# Check Ollama service status
ollama ps

# Start Ollama service
ollama serve

# Test model
ollama run MODEL_NAME "Hello, how are you?"

# Check available models on Ollama library
curl -s https://ollama.com/api/tags | jq '.models[] | .name' | head -20
```

### HuggingFace Models (Embedding/Reranker)
```bash
# List cached HuggingFace models
find ~/.cache/huggingface/hub -name "models--*" -type d | sort

# Check model sizes
du -sh ~/.cache/huggingface/hub/models--*

# Clear HuggingFace cache (use with caution)
# rm -rf ~/.cache/huggingface/

# Remove specific models and lock files
rm -rf ~/.cache/huggingface/hub/models--BAAI--*
rm -rf ~/.cache/huggingface/hub/models--sentence-transformers--*
rm -rf ~/.cache/huggingface/hub/.locks/models--*
rm -rf /Users/kita/SARA/model_cache/huggingface

# Download embedding models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Download reranker models
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-large')"

# Download model via transformers (generic)
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('MODEL_NAME'); AutoTokenizer.from_pretrained('MODEL_NAME')"

# Check model info
python -c "from transformers import AutoConfig; config = AutoConfig.from_pretrained('MODEL_NAME'); print(config)"
```

### spaCy Models
```bash
# List installed spaCy models
python -c "import spacy; print('Installed spaCy models:'); [print(f'  {model}') for model in spacy.util.get_installed_models()]"

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md  
python -m spacy download en_core_web_lg

# Check model info
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print(nlp.meta)"

# Test model
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); doc = nlp('Hello world'); print([(token.text, token.pos_) for token in doc])"

# Remove spaCy models
pip uninstall en_core_web_sm en_core_web_md en_core_web_lg --yes
```

### Current System Configuration
```bash
# Current Ollama models (as of last check)
# qwen3:14B (9.3 GB) - Large context model
# qwen3:8b (5.2 GB) - Balanced performance
# llama3.1:8b (4.9 GB)
# qwen2.5:7b-instruct (4.7 GB)
# qwen2.5:3b-instruct (1.9 GB) - Currently configured
# qwen2.5-coder:7b-instruct-q8_0 (8.1 GB)
# deepseek-r1:8b (4.9 GB)
# deepseek-r1:1.5b (1.1 GB)

# Current HuggingFace models
# BAAI/bge-base-en-v1.5 - Base embedding
# BAAI/bge-large-en-v1.5 - Large embedding (currently configured)
# BAAI/bge-reranker-base - Base reranker (currently configured)  
# BAAI/bge-reranker-large - Large reranker
# sentence-transformers/all-MiniLM-L6-v2 - Lightweight embedding

# Current spaCy models
# en_core_web_sm - Small English model (12MB)
# en_core_web_md - Medium English model (40MB) 
# en_core_web_lg - Large English model (560MB)
```

## Environment and Configuration

### Environment Files
```bash
# Check current environment
echo $SARA_ENV

# View local config
cat .env.local

# View production config  
cat .env.production

# Switch environment
export SARA_ENV=production  # or local

# Reload config and restart
python main.py
```

### Python Environment
```bash
# Check Python version
python --version

# Check pip packages
pip list | grep -E "(torch|transformers|sentence|ollama|chromadb|langchain)"

# Check virtual environment
which python
echo $VIRTUAL_ENV

# Install/update key packages
pip install --upgrade torch torchvision torchaudio
pip install --upgrade transformers sentence-transformers
pip install --upgrade langchain langchain-ollama
pip install --upgrade chromadb
```

## Performance Monitoring

### Resource Usage
```bash
# Real-time system monitor
top -o cpu

# Memory usage by process
ps aux --sort=-%mem | head

# Python process monitoring
ps aux | grep python

# Disk I/O
iostat -w 2

# Network activity
nettop -s 2
```

### ML Workload Monitoring
```bash
# Monitor GPU usage (if available)
sudo powermetrics --samplers gpu_power -n 1

# Temperature monitoring during ML tasks
while true; do sudo powermetrics --samplers smc -n 1 2>/dev/null | grep -i temp | head -3; sleep 2; done

# Memory pressure
memory_pressure

# Watch log files
tail -f logs/sara.log
```

## Development and Testing

### Model Testing Commands
```bash
# Quick LLM test
ollama run qwen2.5:3b-instruct "Explain quantum computing in one sentence"

# Embedding test
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
embeddings = model.encode(['Hello world', 'How are you?'])
print(f'Embedding shape: {embeddings.shape}')
"

# Performance benchmark
time ollama run qwen2.5:3b-instruct "Count from 1 to 10"

# Memory usage test
/usr/bin/time -l ollama run qwen2.5:3b-instruct "Hello"
```

### Sara Application Commands
```bash
# Run Sara
python main.py

# Run with specific environment
SARA_ENV=production python main.py

# Force reindex
SARA_FORCE_REINDEX=True python main.py

# Debug mode
SARA_LOG_LEVEL=DEBUG python main.py

# Check Sara configuration
python -c "from config import config; print(f'Model: {config.LLM_MODEL_NAME}'); print(f'Embedding: {config.EMBEDDING_MODEL_NAME}'); print(f'Environment: {config.ENV}')"
```

## Useful ML/LLM Commands

### Model Comparison and Selection
```bash
# Compare model sizes
ollama list | awk '{print $1 "\t" $3}' | column -t

# Test multiple models quickly
for model in qwen2.5:3b-instruct qwen3:8b qwen3:14b; do
  echo "Testing $model:"
  time ollama run $model "2+2=" --verbose
  echo "---"
done

# Benchmark embedding models
python -c "
import time
from sentence_transformers import SentenceTransformer

models = ['BAAI/bge-base-en-v1.5', 'BAAI/bge-large-en-v1.5']
test_text = 'This is a test sentence for embedding benchmark.'

for model_name in models:
    model = SentenceTransformer(model_name)
    start = time.time()
    embedding = model.encode(test_text)
    end = time.time()
    print(f'{model_name}: {end-start:.3f}s, shape: {embedding.shape}')
"
```

### Advanced Configuration
```bash
# Check available Ollama models online
curl -s https://ollama.com/api/tags | jq -r '.models[] | "\(.name) - \(.size)"' | head -20

# Monitor model download progress
watch -n 1 'du -sh ~/.ollama/models/*'

# Clean up unused Docker containers (if using containerized setup)
docker system prune -f

# Check ChromaDB status
python -c "
import chromadb
client = chromadb.Client()
collections = client.list_collections()
print(f'Collections: {[c.name for c in collections]}')
"

# Backup vector store
cp -r vector_store vector_store_backup_$(date +%Y%m%d_%H%M%S)
```

## Troubleshooting

### Common Issues
```bash
# Ollama not responding
killall ollama
ollama serve

# Check if ports are in use
lsof -i :11434

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Reset HuggingFace cache permissions
chmod -R 755 ~/.cache/huggingface/

# Check disk space (models take lots of space)
df -h
du -sh ~/.ollama ~/.cache/huggingface

# Memory issues - check swap
sysctl vm.swapusage

# Check for memory leaks
leaks -atExit -- python main.py
```

### Performance Optimization
```bash
# Optimize for Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Set optimal thread counts
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Monitor thermal throttling
pmset -g thermlog

# Check energy impact
powerstats -n 10 -s 1
```

---

## System Specifications

**Hardware**: MacBook Pro M2 Pro, 32GB RAM  
**Operating System**: macOS 15.6  
**Last Updated**: August 29, 2025

