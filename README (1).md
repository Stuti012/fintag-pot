# Financial QA System with Program-of-Thought Reasoning

A research-grade Financial Question Answering system that implements:
- **Program-of-Thought (PoT)** reasoning for FinQA benchmark
- **Table-Aware Retrieval-Augmented Generation (RAG)** for both FinQA and SEC EDGAR filings
- **Hybrid Retrieval** (BM25 + Vector search)
- **Temporal Decay** for time-sensitive financial data
- **Numeric Verification** to prevent hallucinations

## Architecture

### Core Components

1. **FinQA Mode** (Benchmark Evaluation)
   - Evidence Retriever: Hybrid BM25 + Vector search with table-aware indexing
   - Program Generator: Generates executable programs in FinQA DSL
   - Program Executor: Deterministic execution in Python
   - Verifier: Ensures all numbers come from evidence

2. **EDGAR Mode** (Real-World Application)
   - EDGAR Downloader: Fetches 10-K/10-Q filings from SEC
   - HTML-to-Markdown Converter: Preserves tables
   - Section Splitter: Extracts Items 1, 1A, 7, 8, etc.
   - Table Parser: Separate indexing for financial tables
   - Temporal Decay Retriever: Prioritizes recent filings
   - Answer Generator with Citations

3. **Verification**
   - Numeric Verifier: Cross-checks all numbers against evidence
   - XBRL Verifier (optional): Validates against structured XBRL data

## Installation

### 1. Clone and Setup

```bash
cd finrag_pot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/check_env.py
```

Optional (higher-quality semantic embeddings + ChromaDB persistence):

```bash
python -m pip install -r requirements-optional.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required in `.env`:
```
ANTHROPIC_API_KEY=your_key_here
# OR
OPENAI_API_KEY=your_key_here

EDGAR_USER_AGENT=YourName your.email@example.com
```


If optional dependencies are not installed, the app still runs using a built-in lightweight hashing embedding fallback and a NumPy vector store.

### 3. Download FinQA Dataset

Download from: https://github.com/czyssrs/FinQA

```bash
mkdir -p data/finqa
# Place train.json, validation.json, test.json in data/finqa/
```

## Troubleshooting

If `app.py` fails very early with missing package errors, run the preflight checker:

```bash
python scripts/check_env.py
```

If you see this error:

```python
import yaml
ModuleNotFoundError: No module named 'yaml'
```

Install project dependencies in your active environment:

```bash
python -m pip install -r requirements.txt
python scripts/check_env.py
```

Or install only YAML support quickly:

```bash
python -m pip install pyyaml
```

The default setup avoids `chromadb` so it works on Windows/Python 3.11+ without Rust toolchain issues:

```bash
python -m pip install -r requirements.txt
python scripts/check_env.py
```

`HybridRetriever` now uses a built-in NumPy vector store by default and still supports ChromaDB automatically when it is installed.

If installation fails with messages involving `ninja`, `cmake`, or SSL certificate verification (common on restricted Windows environments), use the base requirements only:

```bash
python -m pip install -r requirements.txt
```

Then install optional dependencies later when certificate/toolchain access is available.

If you see this error:

```python
ModuleNotFoundError: No module named 'pydantic'
```

Install dependencies into the same Python interpreter used to run the app:

```bash
python -m pip install -r requirements.txt
python scripts/check_env.py
```

Or install only pydantic quickly:

```bash
python -m pip install pydantic
```

You can verify installation with:

```bash
python -c "import yaml, pydantic; print('ok')"
```

## Usage

### Mode 1: FinQA Benchmark Evaluation

#### Run Single Query

```bash
# First, the system needs to index the question's document
python app.py --mode finqa \
  --query "What was the percentage increase in revenue?" \
  --question-id "QUESTION_ID_FROM_DATASET"
```

#### Run Full Evaluation

```bash
# Evaluate on validation set (multiline)
python app.py --mode eval \
  --eval-split validation \
  --num-examples 100

# Or one-line (safer to copy on Windows terminals)
python app.py --mode eval --eval-split validation --num-examples 100
```

This will:
1. Index all examples
2. Run retrieval + program generation + execution
3. Compute metrics:
   - Exact Match
   - Numerical Accuracy
   - Program Execution Success Rate
   - Evidence Precision
   - Hallucination Rate

Results saved to: `./evaluation_results/`

### Mode 2: SEC EDGAR Real-World Queries

#### Download Filings

```bash
python app.py --mode edgar \
  --download \
  --ticker AAPL
```

Downloads recent 10-K and 10-Q filings for Apple.

#### Build Search Index

```bash
python app.py --mode edgar \
  --build-index \
  --ticker AAPL
```

This will:
1. Convert HTML to Markdown
2. Split into sections (Items)
3. Extract and index tables separately
4. Create table summaries for better retrieval
5. Build hybrid search index

#### Query EDGAR

```bash
python app.py --mode edgar \
  --query "What was Apple's revenue in fiscal 2023?" \
  --ticker AAPL
```

Output includes:
- Answer with "As of [DATE]"
- Citations (filing type, date, section)
- Numeric verification results

### Mode 3: Baseline vs Proposed Comparison

```bash
python app.py --mode compare \
  --eval-split validation \
  --num-examples 50
```

Compares:
- **Baseline**: Text-only, vector-only, direct generation
- **Proposed**: Table-aware, hybrid retrieval, PoT reasoning

Outputs:
- `baseline_vs_proposed.json`: Detailed metrics
- `baseline_vs_proposed.png`: Visualization
- Shows improvement percentages

## System Design

### Program-of-Thought (PoT) Pipeline

```
Question → Retrieval → Program Generation → Program Execution → Answer
           ↓            ↓                      ↓
        Evidence     Verify Numbers       Deterministic
        (Text +      from Evidence         Execution
         Tables)
```

### Allowed Operations

```python
add(a, b)         # Addition
subtract(a, b)    # Subtraction
multiply(a, b)    # Multiplication
divide(a, b)      # Division
percent(a, b)     # a as percentage of b
diff(a, b)        # Absolute difference
max(list)         # Maximum value
min(list)         # Minimum value
sum(list)         # Sum of list
avg(list)         # Average of list
```

### Example Program

```python
const_0 = 100.5    # Revenue 2023
const_1 = 85.3     # Revenue 2022
result_0 = subtract(const_0, const_1)  # Difference
result_1 = divide(result_0, const_1)   # Growth rate
answer = multiply(result_1, 100)       # Percentage
```

### Table-Aware Indexing

For each document:
1. **Raw Table** → Indexed as Markdown
2. **Table Summary** → LLM-generated summary for retrieval
3. **Text Chunks** → Surrounding context
4. **Combined** → All indexed together for hybrid search

### Temporal Decay Formula

```
S_final = S_semantic × e^(-λ(t_now - t_doc))
```

Where:
- `S_semantic`: Base retrieval score
- `λ`: Decay rate (default: 0.1)
- `t_now - t_doc`: Age of document in years

Ensures recent filings are prioritized.

## Configuration

Edit `config.yaml` to customize:

```yaml
llm:
  provider: anthropic  # or openai
  model: claude-sonnet-4-20250514
  temperature: 0.0

retrieval:
  top_k: 5
  bm25_weight: 0.5      # Baseline: 0.0
  vector_weight: 0.5     # Baseline: 1.0
  enable_hybrid: true    # Baseline: false

indexing:
  table_summary_enabled: true  # Baseline: false

temporal_decay:
  lambda: 0.1
  enabled: true

verification:
  enable_numeric_verification: true
  tolerance: 0.01
```

## Evaluation Metrics

### FinQA Metrics

- **Exact Match**: Exact numerical match with gold answer
- **Numerical Accuracy**: Match within tolerance (default: 0.001)
- **Execution Success Rate**: % of programs that execute without error
- **Evidence Precision**: % of gold answers found in retrieved evidence
- **Hallucination Rate**: % of numbers in programs not found in evidence

### EDGAR Metrics

- **Citation Coverage**: All facts properly cited
- **Numerical Consistency**: Numbers match retrieved data
- **Retrieval Accuracy**: Table hit rate for table-based queries
- **Temporal Correctness**: Most recent filing used when appropriate

## Project Structure

```
finrag_pot/
├── config.yaml              # System configuration
├── requirements.txt         # Dependencies
├── .env.example            # Environment template
├── app.py                  # Main application
├── finqa/                  # FinQA components
│   ├── load_finqa.py
│   ├── finqa_retriever.py
│   ├── program_generator.py
│   ├── program_executor.py
│   └── finqa_eval.py
├── edgar/                  # EDGAR components
│   ├── edgar_download.py
│   ├── html_to_md.py
│   ├── section_splitter.py
│   └── table_parser.py
├── indexing/              # Indexing system
│   ├── schema.py
│   └── build_index.py
├── retrieval/             # Retrieval engines
│   ├── hybrid_retriever.py
│   └── temporal_decay.py
├── llm/                   # LLM interface
│   ├── llm_client.py
│   └── prompts.py        # All prompts defined here
├── verification/          # Verification modules
│   ├── numeric_verifier.py
│   └── xbrl_verifier.py
└── evaluation/           # Evaluation
    └── compare_baseline_vs_proposed.py
```

## Research Contributions

### Key Innovations

1. **Program-of-Thought for Financial QA**
   - Forces LLM to show work via executable programs
   - Prevents hallucination by requiring evidence grounding
   - Provides interpretable reasoning chain

2. **Table-Aware Hierarchical Indexing**
   - Separates tables from text for targeted retrieval
   - Generates semantic summaries for better matching
   - Preserves table structure for extraction

3. **Hybrid Retrieval with Temporal Decay**
   - Combines keyword matching (BM25) with semantic search
   - Time-weighted scoring for financial documents
   - Adaptive to query type (factual vs analytical)

4. **Multi-Stage Verification**
   - Pre-execution: Verify numbers exist in evidence
   - Post-execution: Cross-check results
   - Optional: XBRL structured data validation

## Example Outputs

### FinQA Query

```
Question: What is the change in operating lease liabilities from 2018 to 2019?

Program:
const_0 = 1003.1
const_1 = 875.8
result_0 = subtract(const_0, const_1)
answer = result_0

Answer: 127.3

Evidence:
- Table showing operating lease liabilities
- 2019: $1,003.1 million
- 2018: $875.8 million
```

### EDGAR Query

```
Query: What was Microsoft's revenue growth in fiscal 2023?

Answer: As of July 2023, Microsoft reported total revenue of $211.9 billion 
for fiscal year 2023, representing an increase of 6.9% compared to $198.3 
billion in fiscal year 2022 [MSFT-10K-2023-07-30:Item7_chunk_0].

Citations:
- MSFT 10-K (2023-07-30) - Item 7: MD&A
- MSFT 10-K (2023-07-30) - Item 8: Financial Statements

Verification: ✓ All numbers verified in source documents
```

## Troubleshooting

If `app.py` fails very early with missing package errors, run the preflight checker:

```bash
python scripts/check_env.py
```

### ChromaDB Issues
```bash
rm -rf ./chroma_db  # Clear existing database
# Rerun indexing
```

### Out of Memory
- Reduce `chunk_size` in config.yaml
- Reduce `num_examples` for evaluation
- Use smaller embedding model

### Rate Limits
- Adjust `temperature` and retry
- Enable caching in LLM client
- Reduce concurrent requests

## Citation

If you use this system, please cite:

```bibtex
@software{finrag_pot_2024,
  title = {Financial QA System with Program-of-Thought Reasoning},
  author = {Your Name},
  year = {2026},
  note = {Research implementation for FinQA and SEC EDGAR}
}
```


## References

- FinQA Dataset: https://github.com/czyssrs/FinQA
- SEC EDGAR: https://www.sec.gov/edgar
- Program-of-Thought: Chen et al., 2022
- RAG: Lewis et al., 2020
