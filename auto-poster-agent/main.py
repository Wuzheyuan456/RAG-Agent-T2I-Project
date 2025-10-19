# main.py
from fastapi import FastAPI, UploadFile
from orchestrator import run
import   uvicorn
app = FastAPI()


from pydantic import BaseModel

class QueryIn(BaseModel):
    q: str

@app.post("/generate")
def generate(body: QueryIn):
    final_path = run(body.q)
    return {"download_url": f"/download/{final_path.split('/')[-1]}"}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)