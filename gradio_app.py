import gradio as gr
import requests
import io
import base64
import numpy as np
import time
from PIL import Image

API_BASE = "http://localhost:1504"

def gradio_process_image(image, detection_prompt, background_prompt):
    """
    This function sends the image and prompts to the FastAPI /process_image/ endpoint,
    polls the /get_result/ endpoint until processing is complete, and returns the processed image.
    """
    # Convert numpy image to PNG bytes
    buffered = io.BytesIO()
    Image.fromarray(image).save(buffered, format="PNG")
    buffered.seek(0)

    # Send POST request to /process_image/ endpoint
    response = requests.post(
        f"{API_BASE}/process_image/",
        files={"file": ("image.png", buffered, "image/png")},
        data={"detection_prompt": detection_prompt, "background_prompt": background_prompt}
    )
    
    if response.status_code != 200:
        return None, f"Error: {response.text}"

    task_id = response.json().get("task_id")
    if not task_id:
        return None, "No task_id received."

    # Poll the /get_result/ endpoint until the task completes or times out
    timeout = 200# seconds
    start_time = time.time()
    status = "Processing..."
    
    while True:
        res = requests.get(f"{API_BASE}/get_result/", params={"task_id": task_id})
        result_json = res.json()
        
        if result_json.get("status") == "completed":
            base64_image = result_json.get("result_image")
            image_bytes = base64.b64decode(base64_image)
            processed_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            processed_image = np.array(processed_image)
            return processed_image, "Success"
        elif result_json.get("status") == "failed":
            error_msg = result_json.get("error", "Unknown error")
            return None, f"Processing failed: {error_msg}"
        elif time.time() - start_time > timeout:
            return None, "Timeout waiting for result."
        
        time.sleep(2)

# Create the Gradio interface
iface = gr.Interface(
    fn=gradio_process_image,
    inputs=[
        gr.Image(label="Input Image", type="numpy"),
        gr.Textbox(label="Detection Prompt", placeholder="Enter an object description (e.g., 'a cat')"),
        gr.Textbox(label="Background Prompt", placeholder="Enter a background description (e.g., 'a cat in a jungle')")
    ],
    outputs=[
        gr.Image(label="Processed Image", type="numpy"),
        gr.Textbox(label="Status")
    ],
    title="Prompt based environement transformation",
    description=f"""Upload an image with an object detection prompt and a background transformation prompt to detect the object and modify its environment accordingly."""
)

if __name__ == "__main__":
    iface.launch(server_port=1508, server_name="0.0.0.0", root_path='/image-inpainting-gradio/')
