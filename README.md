# Multi-Agent Resume Screener

Screen resumes against a job description using two concurrent Gemini agents and a final synthesizer. Built with FastAPI, Gradio, and LangChain — every output is validated against a strict Pydantic schema.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.110%2B-009688">
  <img alt="Gradio" src="https://img.shields.io/badge/Gradio-UI-orange">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-Gemini-1C3C3C">
</p>

---

## Why this exists

Recruiters waste hours on first-pass resume reads, and naive single-prompt LLM screeners conflate "has the skill" with "has done the work." This project splits the screening into two specialists that run in parallel, then a synthesizer reconciles them into a single calibrated decision — fast, structured, and consistent enough to feed into a hiring pipeline.

## How it works

```
              ┌────────────────────────────┐
              │  Resume (PDF/TXT)  +  JD   │
              └────────────┬───────────────┘
                           │
              ┌────────────┴───────────────┐
              │   asyncio.gather (parallel)│
              └─────┬───────────────┬──────┘
                    │               │
       ┌────────────▼──────┐  ┌─────▼──────────────┐
       │ Agent 1: Skills & │  │ Agent 2: Experience│
       │   Qualifications  │  │  & Project Relevance│
       └────────────┬──────┘  └─────┬──────────────┘
                    │               │
                    └───────┬───────┘
                            │
                  ┌─────────▼──────────┐
                  │   Synthesizer LLM   │
                  └─────────┬──────────┘
                            │
                ┌───────────▼───────────┐
                │ FinalScreeningResult  │
                │  (Pydantic-validated) │
                └───────────────────────┘
```

1. **Agent 1 — Skills & Qualifications Matcher** extracts required vs. present skills, computes hard / soft / overall skill scores.
2. **Agent 2 — Experience & Project Relevance Evaluator** scores depth, recency, and relevance of work history and projects.
3. Both agents run **concurrently** via `asyncio.gather` — total latency is `max(agent1, agent2)`, not their sum.
4. **Synthesizer** combines both analyses into a `FinalScreeningResult` with:
   - a 0–100 fit score
   - a `Strong Hire` / `Hire` / `Maybe` / `Pass` recommendation
   - a 3–5 sentence recruiter-ready summary
   - bullet lists of top strengths and gaps

All outputs are enforced by Pydantic schemas via LangChain's `with_structured_output` — no fragile string parsing, no malformed JSON.

## Scoring rubric

| Fit score | Recommendation |
| --------- | -------------- |
| 85 – 100  | Strong Hire    |
| 70 – 84   | Hire           |
| 50 – 69   | Maybe          |
| 0 – 49    | Pass           |

The synthesizer defaults to a roughly 50/50 blend of skills and experience, and adjusts when the JD clearly weights one side more heavily.

## Stack

| Layer            | Choice                                                     |
| ---------------- | ---------------------------------------------------------- |
| LLM              | Google Gemini (`gemini-2.5-flash-lite`) via `langchain-google-genai` |
| Orchestration    | LangChain + `asyncio.gather` for parallel agent calls       |
| Backend          | FastAPI + uvicorn (port `8000`)                            |
| Frontend         | Gradio (port `7860`)                                       |
| Resume parsing   | `pypdf` for PDFs; UTF-8 decode for `.txt` / `.md`          |
| Schema / validation | Pydantic v2 with `with_structured_output`              |

## Quickstart

```bash
# 1. Clone
git clone https://github.com/chaitanyaaroraa/resume_checker.git
cd resume_checker

# 2. Add your Gemini API key
cp .env.example .env
# then edit .env and paste your key after GEMINI_API_KEY=
# Get one at https://aistudio.google.com/apikey

# 3. Run everything with one command
python startup.py
```

`startup.py` installs dependencies, starts FastAPI on `:8000`, waits for it to accept connections, then launches the Gradio UI on `:7860`. Ctrl+C cleanly shuts down both.

Open **http://127.0.0.1:7860**, paste a JD, upload a resume (PDF or text), and submit. A typical screening completes in 10–30 seconds.

## API

The backend exposes a small, focused surface:

| Method | Path      | Body                                              | Returns                  |
| ------ | --------- | ------------------------------------------------- | ------------------------ |
| GET    | `/`       | —                                                 | `{status, service}`      |
| GET    | `/health` | —                                                 | `{ok, gemini_key_set}`   |
| POST   | `/screen` | multipart: `job_description` (str), `resume` (file) | `FinalScreeningResult` |

### Example

```bash
curl -X POST http://127.0.0.1:8000/screen \
  -F "job_description=Senior Python Engineer with FastAPI and LLM experience" \
  -F "resume=@./jane_doe.pdf"
```

### Response shape

```jsonc
{
  "fit_score": 87,
  "recommendation": "Strong Hire",
  "summary": "Strong backend engineer with direct LLM experience...",
  "strengths": ["...", "..."],
  "gaps": ["...", "..."],
  "skills_analysis": {
    "required_skills": [...], "present_skills": [...], "missing_skills": [...],
    "hard_skill_score": 90, "soft_skill_score": 80, "overall_skills_score": 88,
    "reasoning": "..."
  },
  "experience_analysis": {
    "relevant_experience": [...], "relevant_projects": [...],
    "depth_score": 85, "recency_score": 90, "relevance_score": 88,
    "overall_experience_score": 87,
    "reasoning": "..."
  }
}
```

Interactive API docs are auto-generated by FastAPI at **http://127.0.0.1:8000/docs**.

## Project layout

```
agent.py        # LangChain agents, prompts, Pydantic schemas, screen_resume()
app.py          # FastAPI app: /screen endpoint, PDF parsing
frontend.py     # Gradio UI that calls the backend
startup.py      # Installs deps, runs backend + frontend together
.env.example    # Template for GEMINI_API_KEY
```

## Configuration

| Variable          | Default                          | Purpose                                                |
| ----------------- | -------------------------------- | ------------------------------------------------------ |
| `GEMINI_API_KEY`  | _(required)_                     | Google AI Studio API key                               |
| `BACKEND_URL`     | `http://127.0.0.1:8000/screen`   | Frontend → backend endpoint (override for remote deploys) |

## Development notes

- `python agent.py` runs a quick self-test against a sample resume + JD — useful for sanity-checking the LLM wiring without spinning up the UI.
- The agent module is fully importable on its own: `from agent import screen_resume`.
- `.env` is gitignored — never commit your API key.
- The frontend talks to the backend over HTTP, so you can host them on different machines by overriding `BACKEND_URL`.

## Troubleshooting

| Symptom                                           | Likely cause                                     | Fix                                                          |
| ------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| `GEMINI_API_KEY is not set`                       | `.env` missing or unread                         | Copy `.env.example` to `.env` and add your key               |
| `Cannot reach backend` in the Gradio UI           | FastAPI not running on `:8000`                   | Run `python startup.py`, or `uvicorn app:app --port 8000`    |
| `Resume text is empty after parsing`              | Scanned-image PDF with no text layer             | Use a text-based PDF, or pre-OCR the file                    |
| Slow first request                                | Gemini cold start                                | Subsequent calls are fast; nothing to do                     |

## License

MIT — see source headers for details.
