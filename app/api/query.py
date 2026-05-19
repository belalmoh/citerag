from fastapi import APIRouter

router = APIRouter()


@router.post("/")
async def query_documents():
    return {"status": "pending", "message": "Query endpoint stub"}
