from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import shutil
from pathlib import Path
import uuid
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
import logging
from datetime import datetime

from compress import H265Compressor

app = FastAPI(title="Video Compression API")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
LOG_DIR = Path("logs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def setup_logging(log_type: str) -> logging.Logger:
    """
    Set up logging to file for single or batch compression.
    
    Args:
        log_type: Either 'single' or 'batch'
    
    Returns:
        Logger instance configured for file output
    """
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"log_{log_type}_{date_str}.txt"
    log_path = LOG_DIR / log_filename
    
    # Create logger
    logger = logging.getLogger(f"{log_type}_compression")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Also add console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Video Compression API is running"}


@app.post("/compress")
async def compress_and_download(file: UploadFile = File(...)):
    logger = setup_logging("single")
    
    # Create a new compressor instance with the logger
    compressor_instance = H265Compressor("config.json", logger=logger)
    
    logger.info(f"=== Single Compression Started ===")
    logger.info(f"Original filename: {file.filename}")
    
    input_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    
    logger.info(f"Saved uploaded file to: {input_path}")
    
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)


    logger.info("Starting golden search optimization...")
    best_crf, best_file, best_score = compressor_instance.golden_search(str(input_path))
    logger.info(f"Golden search completed - Best CRF: {best_crf}, Score: {best_score:.6f}")

    if not best_file:
        logger.error("Golden search failed - no output file produced")
        return JSONResponse(status_code=500, content={"error": "Golden search failed"})

    output_path = OUTPUT_DIR / f"best_{file.filename}"
    shutil.move(best_file, output_path)
    
    logger.info(f"Compressed file saved to: {output_path}")
    logger.info(f"=== Single Compression Completed ===")

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
    logger = setup_logging("batch")
    
    logger.info(f"=== Batch Compression Started ===")
    logger.info(f"Number of files to process: {len(files)}")
    
    def process_file(file: UploadFile):
        # Use the shared logger created at the batch level
        # Create a new compressor instance with the logger
        compressor_instance = H265Compressor("config.json", logger=logger)
        
        input_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        
        logger.info(f"Processing file: {file.filename}")
        
        with open(input_path, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)

        best_crf, best_file, best_score = compressor_instance.golden_search(str(input_path))
        logger.info(f"Golden search for {file.filename} - Best CRF: {best_crf}, Score: {best_score:.6f}")

        if not best_file:
            logger.error(f"Golden search failed for {file.filename}")
            return None

        output_path = OUTPUT_DIR / f"best_{file.filename}"
        shutil.move(best_file, output_path)
        
        return output_path

    output_files = []
    logger.info(f"Starting parallel processing with {min(4, len(files))} workers...")
    
    with ThreadPoolExecutor(max_workers=min(4, len(files))) as executor:
        futures = [executor.submit(process_file, file) for file in files]
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    output_files.append(result)
                    logger.info(f"Successfully processed: {result.name}")
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")

    if not output_files:
        logger.error("All compressions failed")
        return JSONResponse(status_code=500, content={"error": "All compressions failed"})

    logger.info(f"Successfully compressed {len(output_files)} out of {len(files)} files")

    # Create a ZIP archive
    zip_path = OUTPUT_DIR / f"batch_outputs_{uuid.uuid4().hex}.zip"
    logger.info(f"Creating ZIP archive: {zip_path}")
    
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file_path in output_files:
            zipf.write(file_path, arcname=file_path.name)
    
    zip_size = zip_path.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"ZIP archive created - Size: {zip_size:.2f} MB")
    logger.info(f"=== Batch Compression Completed ===")
    
    return FileResponse(
        zip_path,
        filename=zip_path.name,
        media_type="application/zip"
    )


@app.get("/config")
def get_config():
    compressor_instance = H265Compressor("config.json")
    return compressor_instance.config
