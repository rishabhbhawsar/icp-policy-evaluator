# ICP Taxonomy Policy Evaluation Framework

**LLM-as-a-Judge · Asynchronous Python · FastAPI · SQLite · Structured AI Evaluation**

A database-driven backend service that evaluates product and merchant listings against locale-specific Ideal Customer Profile (ICP) taxonomy policies using a structured Large Language Model (LLM) judging pipeline.

The system combines strict Pydantic data contracts, resilient asynchronous model execution, bounded batch concurrency, cryptographic cache invalidation, and non-blocking SQLite persistence to deliver an auditable policy evaluation workflow.

---

## Getting Started

### Prerequisites

- Python 3.10 or newer.
- Git.
- An API key for the configured OpenAI model provider.
- Windows, Linux, or macOS environment with Python virtual environment support.

### 1. Clone the Repository

```bash
git clone https://github.com/rishabhbhawsar/icp-policy-evaluator.git
cd icp-policy-evaluator
```

### 2. Create the Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a local `.env` file containing the required application configuration (e.g., `OPENAI_API_KEY=your_api_key_here`). Never commit API credentials or other secrets to the repository.

### 5. Start the Application

```bash
uvicorn src.main:app --reload
```

The application starts through the FastAPI composition root, using an asynchronous lifespan manager to initialize database connections and handle cleanup.

### 6. Open API Documentation

Access the interactive OpenAPI documentation at `http://127.0.0.1:8000/docs`.

---

## Configuration & Architecture Overview

The application separates runtime configuration from source code through environment-based settings and features an asynchronous SQLite persistence layer using Write-Ahead Logging (WAL) and bounded batch concurrency (`asyncio.Semaphore(10)`). For full architectural details, complete code patterns, and advanced engineering decisions, please refer to the referenced repository documentation.

---

## Author

**Rishabh Bhawsar**

- GitHub: [rishabhbhawsar](https://github.com/rishabhbhawsar)
- LinkedIn: [Rishabh Bhawsar](https://www.linkedin.com/in/rishabh-bhawsar-409098262/)
- Email: rishabhbhawsar53@gmail.com