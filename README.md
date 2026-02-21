#  Automated Real-Time Misinformation Detection, Analysis & Correction System

> **Thesis Project** — *Development of an Automated System for Real-Time Detection, Analysis, and Correction of Misinformation*

![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)


---

##  Overview

This repository contains the research and implementation for a thesis project focused on building an **end-to-end automated pipeline** for detecting, analyzing, and correcting misinformation in real time.

The system combines **Natural Language Processing (NLP)**, **machine learning classification models**, and **automated fact-checking** techniques to identify fake news at scale — processing articles, claims, and social media content as they emerge.

> ⚠️ **Note:** This project is actively under development. Features, modules, and documentation will be updated regularly as the thesis progresses.

---

##  Research Objectives

- Design and implement a real-time misinformation detection pipeline
- Evaluate and compare multiple classification approaches (traditional ML, deep learning, LLM-based)
- Develop an automated correction/contextualization mechanism for flagged content
- Analyze system performance across different news domains and languages

---

##  Repository Structure

```
FakeNewsDetectionSystems/
│
├── src/                    # Core source code
│   ├── ...                 # Detection, analysis & correction modules
│
├── data/                   # Datasets (raw, processed, samples)
│
├── configs/                # Configuration files for models and pipeline
│
├── reports/                # Experiment results, metrics, and analysis reports
│
├── run_pipeline.py         # Main entry point — runs the full pipeline
├── fix_convert.py          # Data format conversion utility
├── setup.py                # Package setup
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/sonamansuryan/FakeNewsDetectionSystems.git
cd FakeNewsDetectionSystems

# Install dependencies
pip install -r requirements.txt

# Install as a package (optional)
pip install -e .
```

---

##  Usage

### Run the Full Pipeline

```bash
python run_pipeline.py
```

Additional configuration options can be set in the `configs/` directory.

---

##  System Architecture

The system is structured as a multi-stage pipeline:

```
Input (Article / Claim / URL)
        │
        ▼
  ┌─────────────┐
  │Preprocessing│  — Tokenization, cleaning, metadata extraction
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Detection  │  — ML/LLM-based fake news classification
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Analysis   │  — Claim extraction, source credibility scoring
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Correction │  — Fact-checking, context generation, flagging
  └─────────────┘
```

---

##  Current Progress

| Module | Status |
|---|---|
| Data collection & preprocessing | 🔄 In Progress |
| Baseline classification models | 🔄 In Progress |
| Deep learning / LLM integration | 🔄 In Progress |
| Real-time processing pipeline | 🔄 In Progress |
| Correction & contextualization | ⏳ Planned |
| Evaluation & benchmarking | ⏳ Planned |
| Web interface / API | ⏳ Planned |

---

##  Datasets

Details about the datasets used for training and evaluation will be documented here as the project evolves.

---

##  Results & Evaluation

Experiment results, model comparisons, and evaluation metrics are available in the [`reports/`](./reports/) directory and will be updated regularly.

---

## Tech Stack

- **Language:** Python
- **ML/NLP:** *(to be detailed as modules are finalized)*
- **Pipeline orchestration:** Custom `run_pipeline.py`

---

##  Citation

If you reference this work, please cite:

```
@mastersthesis{mansuryan2025fakenews,
  author  = {Mansuryan, Sona},
  title   = {Development of an Automated System for Real-Time Detection, Analysis, and Correction of Misinformation},
  year    = {2026},
  school  = {National Polytechnic University of Armenia}
}
```

---

## Contact

**Sona Mansuryan**  
GitHub: [@sonamansuryan](https://github.com/sonamansuryan)

---

*This project is part of an ongoing thesis and is subject to frequent updates.*