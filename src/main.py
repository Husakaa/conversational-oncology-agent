from fastapi import FastAPI
from src.api.routes import router

# Inicializar la app
app = FastAPI(
    title="Agente Conversacional Oncológico",
    description="API híbrida para extracción determinista y síntesis generativa.",
    version="1.0.0"
)

# Añadir el enrutador que contiene los endpoints
app.include_router(router, prefix="/api/v1")

@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "ok", "mensaje": "Servidor Oncológico Operativo. Ve a /docs para interactuar."}