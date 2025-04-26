from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.database import engine
from app.models import Base
from app.routers import auth, config

# Create all database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Freqtrade Multi-tenant API",
    description="FastAPI backend for multi-user Freqtrade deployments",
    version="1.0.0"
)

# Configure CORS (adjust allow_origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to attach workspace info based on X-Username header
@app.middleware("http")
async def attach_workspace(request: Request, call_next):
    username = request.headers.get("X-Username")
    if username:
        # Store username in request.state for downstream dependencies
        request.state.username = username
    response = await call_next(request)
    return response

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Freqtrade Multi-tenant API is running"}

# Include authentication and config routers
app.include_router(auth.router)
app.include_router(config.router)
