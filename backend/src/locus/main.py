from fastapi import FastAPI

app = FastAPI(title="Locus Backend")


@app.get("/health")
async def health():
    return {"status": "ok"}
