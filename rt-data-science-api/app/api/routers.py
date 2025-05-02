from fastapi import APIRouter
from api.endpoints import router as endpoints_router

router = APIRouter()

router.include_router(endpoints_router, prefix="/movies", tags=["movies"])
