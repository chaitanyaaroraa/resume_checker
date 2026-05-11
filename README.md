# Multi-Agent Resume Screener

Screen resumes against a job description using two concurrent Gemini agents and a final synthesizer. Built with FastAPI + Gradio + LangChain.

## How it works

1. **Agent 1 — Skills & Qualifications Matcher**: extracts required vs. present skills, computes hard/soft/overall skill scores.
2. **Agent 2 — Experience & Project Relevance Evaluator**: scores depth, recency, and relevance of work history.
3. Both agents run concurrently via `asyncio.gather`.
4. **Synthesizer**: combines both analyses into a `FinalScreeningResult` with a 0–100 fit score, a Strong Hire / Hire / Maybe / Pass recommendation, a recruiter-ready summary, and bullet lists of strengths and gaps.

All outputs are validated against Pydantic schemas via LangChain's `with_structured_output`.

## Stack

- **LLM**: Google Gemini (`gemini-2.5-flash-lite`) via `langchain-google-genai`
- **Backend**: FastAPI + uvicorn (port 8000)
- **Frontend**: Gradio (port 7860)
- **Resume parsing**: `pypdf` for PDFs, UTF-8 decoding for `.txt` / `.md`

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/chaitanyaaroraa/resume_checker.git
   cd resume_checker
   ```

2. Copy the example env file and add your Gemini API key:
   ```bash
   cp .env.example .env
   # then edit .env and paste your key after GEMINI_API_KEY=
   ```
   Get a key at https://aistudio.google.com/apikey

3. Run everything with one command:
   ```bash
   python startup.py
   ```
   `startup.py` installs dependencies, starts the FastAPI backend on `:8000`, waits for it to come up, then launches the Gradio frontend on `:7860`.

4. Open http://127.0.0.1:7860 in your browser, paste a job description, upload a resume (PDF or text), and submit.

## API

The backend exposes one screening endpoint:

| Method | Path      | Body                                                      | Returns                |
| ------ | --------- | --------------------------------------------------------- | ---------------------- |
| GET    | `/`       | —                                                         | service status         |
| GET    | `/health` | —                                                         | `{ok, gemini_key_set}` |
| POST   | `/screen` | multipart: `job_description` (str), `resume` (file) | `FinalScreeningResult` |

Example:

```bash
curl -X POST http://127.0.0.1:8000/screen \
  -F "job_description=Senior Python Engineer with FastAPI and LLM experience" \
  -F "resume=@./jane_doe.pdf"
```

## Project layout

```
agent.py        # LangChain agents, prompts, Pydantic schemas, screen_resume()
app.py          # FastAPI app: /screen endpoint, PDF parsing
frontend.py     # Gradio UI that calls the backend
startup.py      # Installs deps, runs backend + frontend together
.env.example    # Template for GEMINI_API_KEY
```

## Notes

- `.env` is gitignored — never commit your API key.
- The agent module is importable on its own; `python agent.py` runs a quick self-test against a sample resume + JD.
