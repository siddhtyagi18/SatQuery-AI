from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import UploadedFile

router = APIRouter(prefix="/api/files", tags=["files"])


@router.get("/{file_id}")
def serve_file(file_id: str, db: Session = Depends(get_db)):
    f = db.query(UploadedFile).filter(UploadedFile.id == file_id).first()
    if f is None:
        raise HTTPException(status_code=404, detail="File not found")
    p = Path(f.file_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File on disk missing")
    return FileResponse(p, filename=f.file_name, media_type=f.mime_type or "application/octet-stream")
