from fastapi import APIRouter

router = APIRouter()


@router.post("/")
async def submit_feedback():
    return {"status": "pending", "message": "Feedback endpoint stub"}
