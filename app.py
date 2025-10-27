from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import shutil
from pathlib import Path
import uuid
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile

from compress import H265Compressor

app = FastAPI(title="Video Compression API")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

compressor = H265Compressor("config.json")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Video Compression API is running"}


@app.post("/compress")
async def compress_and_download(file: UploadFile = File(...)):
    input_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    best_crf, best_file, best_score = compressor.golden_search(str(input_path))

    if not best_file:
        return JSONResponse(status_code=500, content={"error": "Golden search failed"})

    output_path = OUTPUT_DIR / f"best_{file.filename}"
    shutil.move(best_file, output_path)

    return FileResponse(
        output_path,
        filename=output_path.name,
        headers={
            "X-Best-CRF": str(best_crf),
            "X-Best-Score": str(best_score)
        }
    )


@app.post("/batch_compress")
async def batch_compress(files: List[UploadFile] = File(...)):
    """
    Compress multiple videos using golden search in parallel.
    Returns a ZIP file containing all compressed outputs.
    """

    def process_file(file: UploadFile):
        input_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        with open(input_path, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)

        best_crf, best_file, best_score = compressor.golden_search(str(input_path))

        if not best_file:
            return None

        output_path = OUTPUT_DIR / f"best_{file.filename}"
        shutil.move(best_file, output_path)
        return output_path

    output_files = []
    with ThreadPoolExecutor(max_workers=min(4, len(files))) as executor:
        futures = [executor.submit(process_file, file) for file in files]
        for f in as_completed(futures):
            try:
                result = f.result()
                if result:
                    output_files.append(result)
            except Exception as e:
                print(f"Error in batch processing: {e}")

    if not output_files:
        return JSONResponse(status_code=500, content={"error": "All compressions failed"})

    # Create a ZIP archive
    zip_path = OUTPUT_DIR / f"batch_outputs_{uuid.uuid4().hex}.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file_path in output_files:
            zipf.write(file_path, arcname=file_path.name)
    
    return FileResponse(
        zip_path,
        filename=zip_path.name,
        media_type="application/zip"
    )


@app.get("/config")
def get_config():
    return compressor.config
