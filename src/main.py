from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.agent import load_index, chat, chat_groq
from typing import List, Dict, Any, Optional
import os
import uuid
import time
import logging
import traceback

logging.basicConfig(level=logging.INFO, filename="error.log", filemode="a")
logger = logging.getLogger("vapi-server")

app = FastAPI()

# Allow all origins for Vapi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Log every request
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f">>> INCOMING: {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
    response = await call_next(request)
    logger.info(f"<<< RESPONSE: {response.status_code} for {request.url.path}")
    return response

# Mount frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Global state
index_data = None
conversation_histories = {}  # session_id -> list of messages

@app.on_event("startup")
def startup_event():
    global index_data
    try:
        index_data = load_index()
        if index_data:
            print(f"Index loaded: {len(index_data['chunks'])} chunks ready.")
        else:
            print("Warning: No index found. Run `python3 src/rag.py` first.")
    except Exception as e:
        print(f"Failed to load index: {e}")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Endpoint for the Web UI"""
    if not index_data:
        return JSONResponse({"response": "System is booting up or index not built yet. Please run `python3 src/rag.py` first."}, status_code=500)
    
    if request.session_id not in conversation_histories:
        conversation_histories[request.session_id] = []
    
    history = conversation_histories[request.session_id]
    
    try:
        response = chat_groq(request.message, index_data, history)
        
        history.append({"role": "user", "content": request.message})
        history.append({"role": "assistant", "content": response})
        
        if len(history) > 20:
            history = history[-20:]
            conversation_histories[request.session_id] = history
        
        return {"response": response}
    except Exception as e:
        return JSONResponse({"response": f"Error: {str(e)}"}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Frontend not found."


# --- OPENAI COMPATIBLE ENDPOINT FOR VAPI ---

class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIRequest(BaseModel):
    messages: List[OpenAIMessage]
    model: str = "custom"
    temperature: float = 0.7
    stream: bool = False

from fastapi.responses import StreamingResponse
import json

@app.post("/v1/chat/completions")
async def vapi_openai_webhook(request: OpenAIRequest):
    """OpenAI compatible endpoint for Vapi Custom LLM"""
    if not index_data:
        return JSONResponse({"error": "Index not loaded"}, status_code=500)
    
    messages_list = request.messages
    if not messages_list:
        return {"choices": [{"message": {"role": "assistant", "content": "No message provided."}}]}
    
    last_user_msg = next((m.content for m in reversed(messages_list) if m.role == "user"), "")
    
    history = []
    for m in messages_list[:-1]:
        if m.role in ["user", "assistant"]:
            history.append({"role": m.role, "content": m.content})
            
    try:
        if not last_user_msg.strip():
            # If Vapi sends an empty message (e.g. noise or initial ping), prevent Gemini API crash
            ai_response = "Hello! I'm Sanjay's AI assistant. How can I help you?"
        else:
            # Run our RAG agent
            ai_response = chat_groq(last_user_msg, index_data, history)
            
        chat_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_time = int(time.time())
        
        if request.stream:
            # Vapi expects a stream
            def generate_stream():
                # Yield role first
                initial_chunk = {
                    "id": chat_id, "object": "chat.completion.chunk", "created": created_time, "model": request.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(initial_chunk)}\n\n"
                
                # Split response into smaller words for smooth streaming
                words = ai_response.split(" ")
                for i, word in enumerate(words):
                    content = word + (" " if i < len(words) - 1 else "")
                    chunk = {
                        "id": chat_id, "object": "chat.completion.chunk", "created": created_time, "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    time.sleep(0.02) # slight delay to mimic typing
                
                # Final chunk
                final_chunk = {
                    "id": chat_id, "object": "chat.completion.chunk", "created": created_time, "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        
        # Non-streaming response
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created_time,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": ai_response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    except Exception as e:
        logger.error(f"Error in chat completions: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

# Duplicate route - Vapi might call without /v1 prefix
@app.post("/chat/completions")
async def vapi_openai_webhook_alt(request: OpenAIRequest):
    """Fallback: same endpoint without /v1 prefix"""
    return await vapi_openai_webhook(request)

# Models endpoint - Vapi may check this
@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "sanjay-ai", "object": "model", "owned_by": "custom"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
