from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from sqlalchemy.orm import Session

from ..config import get_settings
from ..crud import _file_to_uploaded_image
from ..database import get_db
from ..logging_setup import logger
from ..models import UploadedFile
from ..schemas import ImageRole, UploadedImage
from ..services.metadata import unique_stored_name, validate_and_extract_metadata

router = APIRouter(prefix="/api/upload", tags=["upload"])

settings = get_settings()


@router.post("", response_model=UploadedImage)
async def upload_image(
    file: UploadFile = File(...),
    role: ImageRole = Form("single"),
    db: Session = Depends(get_db),
):
    try:
        original_name = file.filename or "unnamed"
        stored_name = unique_stored_name(original_name)
        save_path = settings.upload_dir_path / stored_name

        content = await file.read()
        file_size = len(content)

        with open(save_path, "wb") as f:
            f.write(content)

        try:
            meta = validate_and_extract_metadata(
                file_name=original_name,
                file_path=save_path,
                file_size_bytes=file_size,
                mime_type=file.content_type,
            )
        except ValueError as ve:
            try:
                save_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise HTTPException(status_code=400, detail=str(ve))

        record = UploadedFile(
            file_name=original_name,
            stored_name=stored_name,
            file_path=str(save_path.resolve()),
            file_format=meta["file_format"],
            file_size_bytes=meta["file_size_bytes"],
            mime_type=meta.get("mime_type"),
            modality=meta.get("modality"),
            modality_confidence=meta.get("modality_confidence"),
            acquisition_date=meta.get("acquisition_date"),
            width_px=meta.get("width_px"),
            height_px=meta.get("height_px"),
            band_count=meta.get("band_count"),
            crs=meta.get("crs"),
            gsd_meters=meta.get("gsd_meters"),
            bounds=meta.get("bounds"),
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        logger.info(f"Uploaded file {record.id}: {original_name} ({file_size} bytes) as {meta['file_format']}")
        return _file_to_uploaded_image(record, role)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
