from flask import Flask, render_template, request
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from docx import Document

import os
import io
import json
import re

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing in .env file")

client = Groq(api_key=GROQ_API_KEY)


def extract_text_from_file(uploaded_file):
    filename = uploaded_file.filename.lower()
    file_bytes = uploaded_file.read()

    text = ""

    if filename.endswith(".pdf"):
        pdf = PdfReader(io.BytesIO(file_bytes))
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    elif filename.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"

    elif filename.endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="ignore")

    else:
        raise ValueError("Only PDF, DOCX, and TXT files are supported.")

    return text.strip()


def safe_json_loads(raw_text):
    """
    Tries hard to extract valid JSON from model output.
    """
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Try to find first JSON object inside the text
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


@app.route("/")
def home():
    return render_template("index.html", result=None, raw_json=None, error=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        resume_file = request.files.get("resume")
        role = request.form.get("role", "").strip()
        jobdesc = request.form.get("jobdesc", "").strip()

        if not resume_file or resume_file.filename.strip() == "":
            return render_template(
                "index.html",
                result=None,
                raw_json=None,
                error="Please upload a resume file."
            )

        resume_text = extract_text_from_file(resume_file)

        if not resume_text:
            return render_template(
                "index.html",
                result=None,
                raw_json=None,
                error="Could not extract text from the uploaded file."
            )

        prompt = f"""
You are an expert AI Resume Analyzer.

Task:
Analyze the resume and provide structured career feedback.

Target Role:
{role if role else "Not specified"}

Job Description:
{jobdesc if jobdesc else "Not provided"}

Resume Text:
{resume_text}

Return ONLY valid JSON with this exact structure:

{{
  "overall_score": 0,
  "ats_score": 0,
  "summary": "string",
  "strengths": ["string"],
  "weaknesses": ["string"],
  "missing_keywords": ["string"],
  "suggestions": ["string"],
  "interview_talking_points": ["string"],
  "rewrite_headline": "string",
  "recommended_next_steps": ["string"]
}}

Rules:
- overall_score and ats_score must be integers from 0 to 100
- Keep summary short but useful
- Suggestions should be practical and interview-ready
- Return only JSON, no markdown, no extra text
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional resume analyzer. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_completion_tokens=1200
        )

        raw_content = response.choices[0].message.content or ""
        data = safe_json_loads(raw_content)

        if not data:
            return render_template(
                "index.html",
                result=None,
                raw_json=raw_content,
                error="Model returned invalid JSON."
            )

        # Make sure lists exist so template does not break
        result = {
            "overall_score": data.get("overall_score", 0),
            "ats_score": data.get("ats_score", 0),
            "summary": data.get("summary", ""),
            "strengths": data.get("strengths", []),
            "weaknesses": data.get("weaknesses", []),
            "missing_keywords": data.get("missing_keywords", []),
            "suggestions": data.get("suggestions", []),
            "interview_talking_points": data.get("interview_talking_points", []),
            "rewrite_headline": data.get("rewrite_headline", ""),
            "recommended_next_steps": data.get("recommended_next_steps", [])
        }

        return render_template(
            "index.html",
            result=result,
            raw_json=json.dumps(result, indent=2),
            error=None
        )

    except Exception as e:
        return render_template(
            "index.html",
            result=None,
            raw_json=None,
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)