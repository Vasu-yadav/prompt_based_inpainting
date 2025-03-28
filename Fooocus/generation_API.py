from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
from typing import Optional
from pydantic import BaseModel
import torch
from inpaint_my import InpaintingPipeline
import tempfile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import base64
from random import randint
import cv2

app = FastAPI(
    title="Background Change API",
    description="API for changing image backgrounds using Stable Diffusion XL",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None
OUTPUT_DIR = "output_background_images"

class BackgroundChangeResponse(BaseModel):
    status: str
    base64_image: str
    # output_path: str
    # processing_time: float

@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = InpaintingPipeline()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    try:
        suffix = os.path.splitext(upload_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = upload_file.file.read()
            tmp.write(content)
            return tmp.name
    except Exception:
        raise HTTPException(status_code=500, detail="Could not save uploaded file")

@app.post("/change-background", response_model=BackgroundChangeResponse)
async def change_background(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...)
):
    try:
        if pipeline is None:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
        temp_input_path = save_upload_file_tmp(image)
        temp_mask_path = save_upload_file_tmp(mask)
        
        output_filename = f"output_{os.path.splitext(image.filename)[0]}{os.path.splitext(image.filename)[1]}"
        print(output_filename)
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # if negative_prompt is None:
        #     negative_prompt = "low quality, blurry, distorted, deformed, artificial lighting, unnatural shadows"
        
        import time
        start_time = time.time()
        # seed = randint(0, 2**32 - 1)
        # result = pipeline.change_background(
        #     image_path=temp_input_path,
        #     prompt=prompt,
        #     output_path= output_path,
        #     seed= seed,
        #     save_intermediates=False
        # )
        results, generated_image = pipeline.generate(
            image_path=temp_input_path,
            mask_path=temp_mask_path,
            prompt=prompt,
            negative_prompt="(bad anatomy:1.4, deformed, mutated, malformed:1.4), Accessories, watches, necklaces, shoes, glasses, unreal, poor lighting",
            inpaint_engine="v1",
            inpaint_strength=1.0,
            inpaint_respective_field=0.618
            )
        
        # image.save(output_path)
        cv2.imwrite(output_path, generated_image)
        
        processing_time = time.time() - start_time
        print("before")
        os.unlink(temp_input_path)
        
        with open(output_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        # print(f"after: {encoded_string}")
        return BackgroundChangeResponse(
            status="success",
            base64_image=encoded_string
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/get-result/{filename}")
# async def get_result(filename: str):
#     file_path = os.path.join(OUTPUT_DIR, filename)
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="File not found")
#     return FileResponse(file_path)

@app.on_event("shutdown")
async def shutdown_event():
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1505) #1505 for gradio usage, 1501 for API usage