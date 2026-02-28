from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()

# ✅ CORS (MANDATORY for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = Llama(
    model_path="./models/llama-8b-q4.gguf",
    n_ctx=2048,
    n_threads=6
)

class QuestionRequest(BaseModel):
    role: str
    skills: str
    resume: str
    num_questions: int

@app.post("/generate")
def generate(req: QuestionRequest):
    prompt = f"""
Generate {req.num_questions} interview questions.

Role: {req.role}
Skills: {req.skills}
Resume: {req.resume}

Return ONLY the questions, one per line.
"""

    result = llm(prompt, max_tokens=300, temperature=0.6)
    text = result["choices"][0]["text"]

    questions = [q.strip() for q in text.split("\n") if q.strip()]
    return {"questions": questions}
