from fastapi import APIRouter

router = APIRouter()


@router.post("/")
async def upload_document():
    return {"status": "pending", "message": "Upload endpoint stub"}
