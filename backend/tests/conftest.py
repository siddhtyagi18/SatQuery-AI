import io
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_satquery.db")
os.environ.setdefault("UPLOAD_DIR", "./test_uploads")
os.environ.setdefault("DEBUG", "true")

from app.config import get_settings
_settings = get_settings()
_settings.DATABASE_URL = "sqlite:///./test_satquery.db"
_settings.UPLOAD_DIR = "./test_uploads"

from app.database import Base, engine, get_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


_TEST_DB = BACKEND_ROOT / "test_satquery.db"
if _TEST_DB.exists():
    _TEST_DB.unlink()
_TEST_UPLOADS = BACKEND_ROOT / "test_uploads"
if _TEST_UPLOADS.exists():
    import shutil
    shutil.rmtree(_TEST_UPLOADS, ignore_errors=True)


@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine("sqlite:///./test_satquery.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db(db_engine):
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db):
    from app.main import app

    def _override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture()
def png_file():
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 48), color=(30, 80, 120))
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture()
def sample_png_name():
    return "optical_sample_2024-03-15.png"
