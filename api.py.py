# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import run_all   # your existing logic

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],       # dev; lock down later
  allow_methods=["*"],
  allow_headers=["*"],
)

class ChatRequest(BaseModel):
  message: str

@app.post("/api/chat")
async def chat(req: ChatRequest):
  response = run_all.run(req.message)  # or whatever your API is
  return {"response": response}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
