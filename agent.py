"""
Multi-Agent Resume Screener
---------------------------
Agent 1: Skills & Qualifications Matcher
Agent 2: Experience & Project Relevance Evaluator
Both run concurrently via asyncio.gather, then a final synthesizer
returns a Pydantic-enforced FinalScreeningResult.
"""

import os
import asyncio
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash-lite"


# ---------- Pydantic schemas (structured output) ----------

class SkillsAnalysis(BaseModel):
    required_skills: List[str] = Field(description="Skills/tools/certs required by the JD")
    present_skills: List[str] = Field(description="Skills/tools/certs the candidate has")
    missing_skills: List[str] = Field(description="Required skills the candidate is missing")
    hard_skill_score: int = Field(description="Hard skill alignment 0-100", ge=0, le=100)
    soft_skill_score: int = Field(description="Soft skill alignment 0-100", ge=0, le=100)
    overall_skills_score: int = Field(description="Overall skills match 0-100", ge=0, le=100)
    reasoning: str = Field(description="Brief reasoning for the scores")


class ExperienceAnalysis(BaseModel):
    relevant_experience: List[str] = Field(description="Relevant work experience bullets")
    relevant_projects: List[str] = Field(description="Relevant projects")
    depth_score: int = Field(description="Depth/seniority score 0-100", ge=0, le=100)
    recency_score: int = Field(description="Recency of relevant work 0-100", ge=0, le=100)
    relevance_score: int = Field(description="Relevance to role 0-100", ge=0, le=100)
    overall_experience_score: int = Field(description="Overall experience 0-100", ge=0, le=100)
    reasoning: str = Field(description="Brief reasoning for the scores")


class FinalScreeningResult(BaseModel):
    fit_score: int = Field(description="Final 0-100 fit score", ge=0, le=100)
    recommendation: str = Field(description="Strong Hire / Hire / Maybe / Pass")
    summary: str = Field(description="Concise 3-5 sentence hiring summary for recruiters")
    strengths: List[str] = Field(description="Top 3-5 candidate strengths")
    gaps: List[str] = Field(description="Top 2-4 gaps or concerns")
    skills_analysis: SkillsAnalysis
    experience_analysis: ExperienceAnalysis


# ---------- LLM factory ----------

def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=temperature,
    )


# ---------- Agent 1: Skills & Qualifications Matcher ----------

SKILLS_SYSTEM_PROMPT = """You are Agent 1: a Skills & Qualifications Matcher for resume screening.

Your job:
1. Parse the JOB DESCRIPTION to extract REQUIRED technical skills, tools, frameworks, and certifications (both hard and soft).
2. Parse the RESUME to extract the candidate's PRESENT skills.
3. Compute the set of MISSING required skills.
4. Score:
   - hard_skill_score (technical/tools/certs match) 0-100
   - soft_skill_score (communication, leadership, collaboration) 0-100
   - overall_skills_score 0-100 (weighted blend, hard skills weighted higher for technical roles)
5. Give brief reasoning.

Be objective. If a skill is mentioned with strong evidence of use, count it as present. If it is merely listed without context, weight it lower.
"""


async def skills_agent(resume: str, job_description: str) -> SkillsAnalysis:
    llm = get_llm().with_structured_output(SkillsAnalysis)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SKILLS_SYSTEM_PROMPT),
        ("human", "JOB DESCRIPTION:\n{jd}\n\nRESUME:\n{resume}\n\nProduce the SkillsAnalysis."),
    ])
    chain = prompt | llm
    return await chain.ainvoke({"jd": job_description, "resume": resume})


# ---------- Agent 2: Experience & Project Relevance Evaluator ----------

EXPERIENCE_SYSTEM_PROMPT = """You are Agent 2: an Experience & Project Relevance Evaluator for resume screening.

Your job:
1. Analyze the candidate's work experience and projects from the RESUME.
2. Compare them against the responsibilities and domain in the JOB DESCRIPTION.
3. Score:
   - depth_score (seniority and technical complexity demonstrated) 0-100
   - recency_score (how recent the relevant work is) 0-100
   - relevance_score (alignment with the role's responsibilities and domain) 0-100
   - overall_experience_score 0-100 (weighted blend)
4. List the most relevant experience and projects.
5. Give brief reasoning.

Penalize gaps, irrelevant experience, or stale skills. Reward direct domain matches and end-to-end project ownership.
"""


async def experience_agent(resume: str, job_description: str) -> ExperienceAnalysis:
    llm = get_llm().with_structured_output(ExperienceAnalysis)
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXPERIENCE_SYSTEM_PROMPT),
        ("human", "JOB DESCRIPTION:\n{jd}\n\nRESUME:\n{resume}\n\nProduce the ExperienceAnalysis."),
    ])
    chain = prompt | llm
    return await chain.ainvoke({"jd": job_description, "resume": resume})


# ---------- Synthesizer ----------

SYNTH_SYSTEM_PROMPT = """You are the final hiring synthesizer.

You receive two specialist analyses (skills + experience) and produce a final structured screening report:
- fit_score (0-100): a weighted blend; for most engineering roles use roughly 50% skills + 50% experience, but adjust if the JD weights one heavily.
- recommendation: one of "Strong Hire", "Hire", "Maybe", "Pass".
   - >=85: Strong Hire
   - 70-84: Hire
   - 50-69: Maybe
   - <50: Pass
- summary: 3-5 sentences a busy recruiter can read in 15 seconds.
- strengths: top 3-5 bullets.
- gaps: top 2-4 bullets.
- Include the original skills_analysis and experience_analysis verbatim in your output.

Be honest, calibrated, and concise.
"""


async def synthesize_agent(
    resume: str,
    job_description: str,
    skills: SkillsAnalysis,
    experience: ExperienceAnalysis,
) -> FinalScreeningResult:
    llm = get_llm(temperature=0.1).with_structured_output(FinalScreeningResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTH_SYSTEM_PROMPT),
        ("human", """JOB DESCRIPTION:
{jd}

RESUME:
{resume}

SKILLS ANALYSIS (Agent 1):
{skills}

EXPERIENCE ANALYSIS (Agent 2):
{experience}

Now produce the FinalScreeningResult. Copy the provided skills_analysis and experience_analysis objects into the output unchanged."""),
    ])
    chain = prompt | llm
    return await chain.ainvoke({
        "jd": job_description,
        "resume": resume,
        "skills": skills.model_dump_json(indent=2),
        "experience": experience.model_dump_json(indent=2),
    })


# ---------- Public entry point ----------

async def screen_resume(resume_text: str, job_description: str) -> FinalScreeningResult:
    """
    Run Agent 1 and Agent 2 concurrently, then synthesize the final result.
    Eliminates sequential LLM-call latency.
    """
    skills_result, experience_result = await asyncio.gather(
        skills_agent(resume_text, job_description),
        experience_agent(resume_text, job_description),
    )
    return await synthesize_agent(
        resume_text, job_description, skills_result, experience_result
    )


# Quick CLI test: python agent.py
if __name__ == "__main__":
    sample_resume = """Jane Doe — Senior Python Engineer
    5+ years Python, FastAPI, asyncio, Postgres. Built multi-agent LLM pipelines with LangChain.
    Led migration of monolith to microservices. AWS certified.
    """
    sample_jd = """Looking for a Senior Backend Engineer with Python, FastAPI, async experience,
    LLM/LangChain exposure, and cloud (AWS/GCP) skills. Must own systems end-to-end.
    """
    result = asyncio.run(screen_resume(sample_resume, sample_jd))
    print(result.model_dump_json(indent=2))
