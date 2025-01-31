import os
import cv2
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage, ImageBlock, MessageRole, TextBlock

# Load and Resize Image
img_path = "4.jpg"
image = cv2.imread(img_path)
if image is None:
    raise ValueError(f"Error: Unable to load image from '{img_path}'")
image_resized = cv2.resize(image, (1020, 600))
cv2.imwrite("resized_1.jpg", image_resized)

# Set API Keys
os.environ["GOOGLE_API_KEY"] = ""

# Initialize Models
gemini_pro = Gemini(model_name="models/gemini-1.5-flash")

llm = Groq(
    model="deepseek-r1-distill-llama-70b",
    api_key=""
)

# Step 1: Send Image + Prompt to Gemini
msg_gemini = ChatMessage(
    role=MessageRole.USER,
    blocks=[
        ImageBlock(path="resized_1.jpg", image_mimetype="image/jpeg"),
        TextBlock(text="Detect objects in this image")
    ]
)

response_gemini = gemini_pro.chat(messages=[msg_gemini])

# Extract detected objects text
if response_gemini and response_gemini.message:
    detected_objects = " ".join(
        block.text for block in response_gemini.message.blocks if hasattr(block, "text")
    ).strip()
else:
    raise ValueError("Gemini did not return a valid response.")

# Step 2: Send Gemini's Output to DeepSeek-R1 for Enhancement
msg_deepseek = ChatMessage(
    role=MessageRole.USER,
    blocks=[TextBlock(text=f"Enhance this object detection data: {detected_objects}")]
)

response_llm = llm.chat(messages=[msg_deepseek])

# Print Final Enhanced Response
print(response_llm.message.blocks[0].text if response_llm.message else "No response from DeepSeek-R1")
