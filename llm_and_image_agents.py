from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_ollama import ChatOllama

import cv2
import base64
import ollama

agent_llm = ChatOllama(model="qwen3.5:0.8b")

vision_model = "llava-phi3"

@tool
def capture_camera_image() -> str:
    """Capture a photo from the webcam and return base64 image data."""

    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()

    if not ret:
        return "Failed to capture image."

    _, buffer = cv2.imencode(".jpg", frame)
    image_data = base64.b64encode(buffer).decode("utf-8")

    return image_data


@tool
def analyze_image(image_base64: str) -> str:
    """Analyze an image using the vision model."""

    response = ollama.chat(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": "Describe what you see in this image.",
                "images": [image_base64],
            }
        ],
    )

    return response["message"]["content"]


tools = [capture_camera_image, analyze_image]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a vision-enabled assistant.

To see the environment:
1. Use capture_camera_image
2. Then use analyze_image

Use tools when needed.
""",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(agent_llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)


response = agent_executor.invoke(
    {"input": "Look at the camera and tell me what I'm holding."}
)

print("\nFinal Answer:")
print(response["output"])