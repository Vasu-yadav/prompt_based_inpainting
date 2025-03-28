# Instantiate the image processor
from prompt_inpaint.prompt_inpaint import ImageProcessor
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
import uvicorn
import uuid
import base64
import numpy as np
import io
from PIL import Image

image_processor = ImageProcessor()

# Global dictionary to hold task results keyed by task ID
tasks = {}

app = FastAPI()

def process_image_background(task_id: str, image_bytes: bytes, detection_prompt: str, background_prompt: str):
    # Convert image bytes to a NumPy array (RGB)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image)
    try:
        result_image_array, merged_mask = image_processor.process_image(image_array, detection_prompt, background_prompt)
        # Convert the result image to base64 encoded string
        result_image = Image.fromarray(result_image_array)
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        base64_encoded_result = base64.b64encode(buffered.getvalue()).decode("utf-8")
        tasks[task_id] = {"status": "completed", "result_image": base64_encoded_result}
    except Exception as e:
        tasks[task_id] = {"status": "failed", "error": str(e)}

@app.post("/process_image/")
async def process_image_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    detection_prompt: str = Form(...),
    background_prompt: str = Form(...)
):
    task_id = str(uuid.uuid4())
    file_bytes = await file.read()
    tasks[task_id] = {"status": "processing"}
    # Launch the processing task in the background
    background_tasks.add_task(process_image_background, task_id, file_bytes, detection_prompt, background_prompt)
    return {"task_id": task_id}

@app.get("/get_result/")
async def get_result(task_id: str):
    if task_id not in tasks:
        return {"status": "not found", "message": "Task ID not found."}
    return tasks[task_id]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1504)