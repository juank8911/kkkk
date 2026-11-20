from fastapi import FastAPI
from app.api.routes import ingestion, execution

app = FastAPI(title="OpenClaw API Gateway")

app.include_router(ingestion.router)
app.include_router(execution.router)