"""
FastAPI backend for the Multi-Agent Resume Screener.
Run: uvicorn app:app --host 127.0.0.1 --port 8000
"""

import io
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from dotenv import load_dotenv

from agent import screen_resume, FinalScreeningResult

load_dotenv()

app = FastAPI(title="Multi-Agent Resume Screener", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_text_from_pdf(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


@app.get("/")
def root():
    return {"status": "ok", "service": "Multi-Agent Resume Screener"}


@app.get("/health")
def health():
    return {"ok": True, "gemini_key_set": bool(os.getenv("GEMINI_API_KEY"))}


@app.post("/screen", response_model=FinalScreeningResult)
async def screen(
    job_description: str = Form(...),
    resume: UploadFile = File(...),
):
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(500, "GEMINI_API_KEY is not set in .env")

    content = await resume.read()
    filename = (resume.filename or "").lower()

    try:
        if filename.endswith(".pdf"):
            resume_text = extract_text_from_pdf(content)
        else:
            # .txt, .md, or anything else — decode as text
            resume_text = content.decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(400, f"Failed to parse resume: {e}")

    if not resume_text.strip():
        raise HTTPException(400, "Resume text is empty after parsing.")
    if not job_description.strip():
        raise HTTPException(400, "Job description is empty.")

    try:
        return await screen_resume(resume_text, job_description)
    except Exception as e:
        raise HTTPException(500, f"Screening failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
