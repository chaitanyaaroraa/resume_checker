"""
Gradio frontend for the Multi-Agent Resume Screener.
Run: python frontend.py   (FastAPI must already be running on :8000)
"""

import os
import requests
import gradio as gr

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/screen")


def screen_resume_ui(resume_file, job_description, progress=gr.Progress()):
    if resume_file is None:
        yield "⚠️ Please upload a resume file.", "", "", ""
        return
    if not job_description or not job_description.strip():
        yield "⚠️ Please paste a job description.", "", "", ""
        return

    # Gradio File component returns a filepath string by default in recent versions.
    file_path = resume_file if isinstance(resume_file, str) else getattr(resume_file, "name", None)
    if not file_path or not os.path.exists(file_path):
        yield "❌ Could not read uploaded file.", "", "", ""
        return

    filename = os.path.basename(file_path)

    progress(0.1, desc="Uploading resume...")
    yield "⏳ **Uploading resume and contacting backend...**", "", "", ""

    try:
        with open(file_path, "rb") as f:
            files = {"resume": (filename, f, "application/octet-stream")}
            data = {"job_description": job_description}
            progress(0.3, desc="Running agents in parallel...")
            yield (
                "🤖 **Agents are analyzing the candidate...**\n\n"
                "- Agent 1: Skills & Qualifications\n"
                "- Agent 2: Experience & Project Relevance\n\n"
                "_This usually takes 10–30 seconds._",
                "", "", "",
            )
            response = requests.post(BACKEND_URL, files=files, data=data, timeout=240)
    except requests.exceptions.ConnectionError:
        yield (
            "❌ Cannot reach backend. Is FastAPI running on port 8000?\n\n"
            "Try: `python startup.py` or `uvicorn app:app --port 8000`",
            "", "", ""
        )
        return
    except Exception as e:
        yield f"❌ Request error: {e}", "", "", ""
        return

    if response.status_code != 200:
        yield f"❌ Backend error ({response.status_code}): {response.text}", "", "", ""
        return

    progress(0.9, desc="Formatting results...")
    result = response.json()

    score = result["fit_score"]
    rec = result["recommendation"]

    # Color-code the recommendation
    rec_emoji = {
        "Strong Hire": "🟢",
        "Hire": "🟢",
        "Maybe": "🟡",
        "Pass": "🔴",
    }.get(rec, "⚪")

    score_md = f"# {rec_emoji} Fit Score: **{score}/100**\n### Recommendation: **{rec}**"

    summary_md = f"## 📋 Hiring Summary\n{result['summary']}"

    strengths = "\n".join(f"- {s}" for s in result["strengths"])
    gaps = "\n".join(f"- {g}" for g in result["gaps"])
    sg_md = f"## ✅ Strengths\n{strengths}\n\n## ⚠️ Gaps & Concerns\n{gaps}"

    skills = result["skills_analysis"]
    exp = result["experience_analysis"]

    details_md = f"""## 🔍 Detailed Agent Analysis

### 🛠️ Agent 1 — Skills & Qualifications  ·  Overall: **{skills['overall_skills_score']}/100**
- Hard skills: **{skills['hard_skill_score']}/100**
- Soft skills: **{skills['soft_skill_score']}/100**
- **Present:** {', '.join(skills['present_skills']) or '—'}
- **Missing:** {', '.join(skills['missing_skills']) or '—'}
- *Reasoning:* {skills['reasoning']}

### 💼 Agent 2 — Experience & Project Relevance  ·  Overall: **{exp['overall_experience_score']}/100**
- Depth: **{exp['depth_score']}/100**
- Recency: **{exp['recency_score']}/100**
- Relevance: **{exp['relevance_score']}/100**
- *Reasoning:* {exp['reasoning']}
"""

    progress(1.0, desc="Done")
    yield score_md, summary_md, sg_md, details_md


with gr.Blocks(title="Multi-Agent Resume Screener", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🤖 Multi-Agent Resume Screener\n"
        "Upload a resume (PDF or TXT), paste a job description, and two AI agents "
        "(Skills Matcher + Experience Evaluator) run **in parallel** via `asyncio.gather` "
        "to produce a Pydantic-validated 0–100 fit score."
    )

    with gr.Row():
        with gr.Column(scale=1):
            resume_input = gr.File(
                label="📄 Resume (PDF or TXT)",
                file_types=[".pdf", ".txt"],
                type="filepath",
            )
            jd_input = gr.Textbox(
                label="💼 Job Description",
                lines=14,
                placeholder="Paste the full job description here...",
            )
            submit_btn = gr.Button("🚀 Screen Candidate", variant="primary", size="lg")

        with gr.Column(scale=1):
            score_output = gr.Markdown()
            summary_output = gr.Markdown()
            strengths_gaps_output = gr.Markdown()
            details_output = gr.Markdown()

    submit_btn.click(
        fn=screen_resume_ui,
        inputs=[resume_input, jd_input],
        outputs=[score_output, summary_output, strengths_gaps_output, details_output],
        show_progress="full",
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
