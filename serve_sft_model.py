#!/usr/bin/env python3
"""
Dynamic Model Inference Server with vLLM Backend

Supports loading HuggingFace models on-demand via API calls with vLLM for high-performance inference.
An external webapp can:
1. List available/loaded models
2. Load a new model by HF name
3. Unload models to free memory
4. Run inference on loaded models (with continuous batching)

Usage:
    python serve_sft_model.py --port 8000
    python serve_sft_model.py --port 8000 --preload qwen3-4b-sft
cd /root/workspace/game_rl_training && source venv/bin/activate && TORCH_COMPILE_DISABLE=1 VLLM_USE_TRITON_FLASH_ATTN=0 python serve_sft_model.py --port 8000 --preload qwen3-4b-sft --gpu-memory-utilization 0.85
API Endpoints:
    POST /v1/models/load        - Load a model by HF name
    POST /v1/models/unload      - Unload a model
    GET  /v1/models             - List loaded models
    GET  /v1/models/status      - Get detailed model status
    POST /v1/chat/completions   - Run inference (OpenAI-compatible)
    GET  /health                - Health check
"""

import argparse
import sys
import logging
import asyncio
import time
import gc
import os
from typing import Dict, Optional, Any, List
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class VLLMModelManager:
    """Manages vLLM models with dynamic loading/unloading"""
    
    def __init__(self, max_models: int = 1, gpu_memory_utilization: float = 0.85):
        self.engines: Dict[str, Any] = {}  # model_name -> {"engine": LLM, "loaded_at": timestamp}
        self.max_models = max_models
        self.gpu_memory_utilization = gpu_memory_utilization
        self.current_model = None
        
        # Predefined model aliases for convenience
        self.model_aliases = {
            "qwen3-4b-sft": "/root/workspace/game_rl_training/Qwen3-4B-Instruct-2507-SFT/checkpoint-5928",
            "qwen3-0.6b-sft": "/root/workspace/game_rl_training/Qwen3-0.6B-SFT/checkpoint-5928",
            "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
            "qwen3-0.6b": "Qwen/Qwen3-0.6B",
            "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
            "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
            "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
            "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        }
    
    def resolve_model_name(self, name: str) -> str:
        """Resolve alias to full model path/name"""
        return self.model_aliases.get(name, name)
    
    def load_model(self, model_name: str, force_reload: bool = False) -> dict:
        """Load a model using vLLM"""
        from vllm import LLM, SamplingParams
        import torch
        
        resolved_name = self.resolve_model_name(model_name)
        
        # Check if already loaded
        if resolved_name in self.engines and not force_reload:
            self.current_model = resolved_name
            logger.info(f"Model {model_name} already loaded, switching to it")
            return {
                "status": "already_loaded",
                "model": model_name,
                "resolved_name": resolved_name
            }
        
        # Unload existing models if at capacity (vLLM typically needs exclusive GPU access)
        if len(self.engines) >= self.max_models or force_reload:
            for existing_model in list(self.engines.keys()):
                self._unload_model_internal(existing_model)
            logger.info("Unloaded existing models to make room")
        
        logger.info(f"Loading model with vLLM: {resolved_name}...")
        start_time = time.time()
        
        try:
            # Create vLLM engine
            engine = LLM(
                model=resolved_name,
                trust_remote_code=True,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=4096,  # Reasonable default, can be adjusted
                dtype="bfloat16",
            )
            
            load_time = time.time() - start_time
            
            self.engines[resolved_name] = {
                "engine": engine,
                "loaded_at": datetime.now().isoformat(),
                "load_time": load_time,
                "alias": model_name if model_name != resolved_name else None,
            }
            self.current_model = resolved_name
            
            logger.info(f"Model {model_name} loaded successfully with vLLM in {load_time:.2f}s")
            
            return {
                "status": "loaded",
                "model": model_name,
                "resolved_name": resolved_name,
                "load_time": load_time,
                "backend": "vllm"
            }
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name} with vLLM: {e}")
            return {
                "status": "error",
                "model": model_name,
                "error": str(e)
            }
    
    def _unload_model_internal(self, model_name: str):
        """Internal unload"""
        import torch
        
        if model_name in self.engines:
            # Delete vLLM engine
            del self.engines[model_name]["engine"]
            del self.engines[model_name]
            
            if self.current_model == model_name:
                self.current_model = next(iter(self.engines.keys()), None)
            
            gc.collect()
            torch.cuda.empty_cache()
    
    def unload_model(self, model_name: str) -> dict:
        """Unload a model to free memory"""
        resolved_name = self.resolve_model_name(model_name)
        
        if resolved_name not in self.engines:
            return {"status": "not_loaded", "model": model_name}
        
        self._unload_model_internal(resolved_name)
        logger.info(f"Model {model_name} unloaded")
        
        return {"status": "unloaded", "model": model_name}
    
    def get_engine(self, model_name: str = None):
        """Get a loaded vLLM engine"""
        resolved_name = self.resolve_model_name(model_name) if model_name else self.current_model
        
        if not resolved_name or resolved_name not in self.engines:
            return None
        
        return self.engines[resolved_name]["engine"]
    
    def generate(self, model_name: str, prompts: List[str], 
                 max_tokens: int = 512, temperature: float = 0.7, 
                 top_p: float = 0.9, stop: List[str] = None) -> List[str]:
        """Generate completions using vLLM"""
        from vllm import SamplingParams
        
        engine = self.get_engine(model_name)
        if engine is None:
            raise ValueError(f"Model {model_name} not loaded")
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.01,  # vLLM doesn't like 0
            top_p=top_p,
            stop=stop,
        )
        
        outputs = engine.generate(prompts, sampling_params)
        
        return [output.outputs[0].text for output in outputs]
    
    def list_models(self) -> dict:
        """List all loaded models"""
        models_info = []
        for name, info in self.engines.items():
            models_info.append({
                "id": info.get("alias") or name,
                "resolved_name": name,
                "loaded_at": info["loaded_at"],
                "load_time": info.get("load_time"),
                "backend": "vllm",
                "is_current": name == self.current_model
            })
        
        return {
            "object": "list",
            "data": models_info,
            "current_model": self.current_model,
            "available_aliases": list(self.model_aliases.keys()),
            "backend": "vllm"
        }
    
    def get_status(self) -> dict:
        """Get detailed status"""
        import torch
        
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                gpu_memory[f"cuda:{i}"] = {
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                    "total_gb": round(total, 2),
                    "free_gb": round(total - reserved, 2)
                }
        
        return {
            "backend": "vllm",
            "loaded_models": len(self.engines),
            "max_models": self.max_models,
            "current_model": self.current_model,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "models": self.list_models()["data"],
            "gpu_memory": gpu_memory,
            "available_aliases": self.model_aliases
        }


# Global model manager
model_manager: Optional[VLLMModelManager] = None


def create_app(manager: VLLMModelManager):
    """Create Flask app with vLLM model manager"""
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
    except ImportError:
        print("❌ Required packages not installed. Install with:")
        print("   pip install flask flask-cors")
        sys.exit(1)
    
    from transformers import AutoTokenizer
    
    app = Flask(__name__)
    CORS(app)  # Enable CORS for external webapp access
    
    # Cache tokenizers separately (vLLM handles its own tokenizer internally but we need it for chat template)
    tokenizer_cache = {}
    
    def get_tokenizer(model_name: str):
        resolved = manager.resolve_model_name(model_name) if model_name else manager.current_model
        if resolved not in tokenizer_cache:
            tokenizer_cache[resolved] = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
        return tokenizer_cache[resolved]
    
    @app.route("/v1/models/load", methods=["POST"])
    def load_model():
        """Load a model by HF name or alias"""
        data = request.json or {}
        model_name = data.get("model")
        force_reload = data.get("force_reload", False)
        
        if not model_name:
            return jsonify({"error": "model name required"}), 400
        
        client_ip = request.remote_addr
        logger.info(f"[LOAD] IP: {client_ip} | Model: {model_name}")
        
        result = manager.load_model(model_name, force_reload=force_reload)
        
        # Pre-cache tokenizer
        if result["status"] in ["loaded", "already_loaded"]:
            try:
                get_tokenizer(model_name)
            except Exception as e:
                logger.warning(f"Failed to cache tokenizer: {e}")
        
        status_code = 200 if result["status"] != "error" else 500
        return jsonify(result), status_code
    
    @app.route("/v1/models/unload", methods=["POST"])
    def unload_model():
        """Unload a model"""
        data = request.json or {}
        model_name = data.get("model")
        
        if not model_name:
            return jsonify({"error": "model name required"}), 400
        
        client_ip = request.remote_addr
        logger.info(f"[UNLOAD] IP: {client_ip} | Model: {model_name}")
        
        # Clear tokenizer cache too
        resolved = manager.resolve_model_name(model_name)
        if resolved in tokenizer_cache:
            del tokenizer_cache[resolved]
        
        result = manager.unload_model(model_name)
        return jsonify(result)
    
    @app.route("/v1/models", methods=["GET"])
    def list_models():
        """List loaded models (OpenAI-compatible)"""
        return jsonify(manager.list_models())
    
    @app.route("/v1/models/status", methods=["GET"])
    def model_status():
        """Get detailed model and GPU status"""
        return jsonify(manager.get_status())
    
    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions():
        """OpenAI-compatible chat completions with vLLM"""
        start_time = time.time()
        
        data = request.json
        messages = data.get("messages", [])
        model_name = data.get("model")
        max_tokens = data.get("max_tokens", 512)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        
        # Get engine (use specified or current)
        resolved_model = manager.resolve_model_name(model_name) if model_name else manager.current_model
        engine = manager.get_engine(model_name)
        
        if engine is None:
            # Try to auto-load if model specified
            if model_name:
                result = manager.load_model(model_name)
                if result["status"] == "error":
                    return jsonify({"error": f"Model not loaded and failed to load: {result.get('error')}"}), 400
                engine = manager.get_engine(model_name)
                resolved_model = manager.resolve_model_name(model_name)
            else:
                return jsonify({"error": "No model loaded. Use /v1/models/load first or specify model in request"}), 400
        
        # Log request
        client_ip = request.remote_addr
        user_message = messages[-1].get("content", "")[:100] if messages else ""
        logger.info(f"[REQUEST] IP: {client_ip} | Model: {model_name or manager.current_model} | Messages: {len(messages)} | MaxTokens: {max_tokens}")
        logger.info(f"[REQUEST] User message (truncated): {user_message}...")
        
        try:
            # Get tokenizer and apply chat template
            tokenizer = get_tokenizer(model_name)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Count input tokens
            input_tokens = len(tokenizer.encode(prompt))
            
            # Generate with vLLM
            responses = manager.generate(
                model_name,
                [prompt],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            response_text = responses[0]
            output_tokens = len(tokenizer.encode(response_text))
            
            elapsed_time = time.time() - start_time
            tokens_per_sec = output_tokens / elapsed_time if elapsed_time > 0 else 0
            
            # Log response
            response_preview = response_text[:200].replace('\n', ' ')
            logger.info(f"[RESPONSE] IP: {client_ip} | Time: {elapsed_time:.2f}s | InTokens: {input_tokens} | OutTokens: {output_tokens} | Speed: {tokens_per_sec:.1f} tok/s")
            logger.info(f"[RESPONSE] Content (truncated): {response_preview}...")
            
            return jsonify({
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name or manager.current_model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                "backend": "vllm"
            })
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/v1/completions", methods=["POST"])
    def completions():
        """OpenAI-compatible text completions with vLLM"""
        start_time = time.time()
        
        data = request.json
        prompt = data.get("prompt", "")
        model_name = data.get("model")
        max_tokens = data.get("max_tokens", 512)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        
        engine = manager.get_engine(model_name)
        if engine is None:
            return jsonify({"error": "No model loaded"}), 400
        
        try:
            responses = manager.generate(
                model_name,
                [prompt] if isinstance(prompt, str) else prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            elapsed_time = time.time() - start_time
            
            return jsonify({
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name or manager.current_model,
                "choices": [
                    {"index": i, "text": text, "finish_reason": "stop"}
                    for i, text in enumerate(responses)
                ],
                "backend": "vllm"
            })
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/health", methods=["GET"])
    def health():
        """Health check"""
        return jsonify({
            "status": "healthy",
            "backend": "vllm",
            "models_loaded": len(manager.engines),
            "current_model": manager.current_model
        })
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Dynamic Model Inference Server with vLLM")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--max-models", type=int, default=1, help="Max models to keep loaded (vLLM typically needs 1)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--preload", type=str, default=None, help="Model to preload on startup")
    
    args = parser.parse_args()
    
    # Create model manager
    global model_manager
    model_manager = VLLMModelManager(
        max_models=args.max_models,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # Preload model if specified
    if args.preload:
        logger.info(f"Preloading model with vLLM: {args.preload}")
        result = model_manager.load_model(args.preload)
        if result["status"] == "error":
            logger.error(f"Failed to preload model: {result.get('error')}")
            sys.exit(1)
    
    # Create and run app
    app = create_app(model_manager)
    
    print(f"\n🚀 Dynamic Model Inference Server (vLLM Backend)")
    print("=" * 60)
    print(f"Host: {args.host}:{args.port}")
    print(f"Backend: vLLM (high-performance)")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Max concurrent models: {args.max_models}")
    print("\nAPI Endpoints:")
    print(f"  POST /v1/models/load      - Load a model")
    print(f"  POST /v1/models/unload    - Unload a model")
    print(f"  GET  /v1/models           - List loaded models")
    print(f"  GET  /v1/models/status    - Detailed status")
    print(f"  POST /v1/chat/completions - Chat (OpenAI-compatible)")
    print(f"  POST /v1/completions      - Text completion")
    print(f"  GET  /health              - Health check")
    print("\nAvailable model aliases:")
    for alias, path in model_manager.model_aliases.items():
        print(f"  {alias:20} -> {path}")
    print("=" * 60 + "\n")
    
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
