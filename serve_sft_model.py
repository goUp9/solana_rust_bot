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

    # If you encounter Triton compilation errors (Python.h not found), use:
    TORCH_COMPILE_DISABLE=1 python serve_sft_model.py --port 8000 --preload qwen3-4b-sft

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

# Fix for Triton compilation errors (Python.h not found)
# This disables torch.compile which requires Python dev headers
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def is_lora_adapter(path: str) -> bool:
    """Check if a path contains a LoRA adapter (has adapter_config.json but no config.json)"""
    import os
    if not os.path.isdir(path):
        return False
    has_adapter_config = os.path.exists(os.path.join(path, "adapter_config.json"))
    has_model_config = os.path.exists(os.path.join(path, "config.json"))
    return has_adapter_config and not has_model_config


def get_lora_base_model(adapter_path: str) -> Optional[str]:
    """Get the base model name from a LoRA adapter config"""
    import json
    import os
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
            return config.get("base_model_name_or_path")
    return None


class VLLMModelManager:
    """Manages vLLM models with dynamic loading/unloading, with PEFT fallback for LoRA"""
    
    def __init__(self, max_models: int = 1, gpu_memory_utilization: float = 0.85):
        self.engines: Dict[str, Any] = {}  # model_name -> {"engine": LLM/model, "loaded_at": timestamp, "backend": "vllm"/"peft"}
        self.max_models = max_models
        self.gpu_memory_utilization = gpu_memory_utilization
        self.current_model = None
        
        # Predefined model aliases for convenience
        self.model_aliases = {
            "qwen3-4b-sft": "/root/workspace/game_rl_training/Qwen3-4B-Instruct-2507-SFT/checkpoint-5928",
            "qwen3-0.6b-sft": "/root/workspace/game_rl_training/Qwen3-0.6B-SFT/checkpoint-5928",
            # LoRA adapter (will use PEFT backend automatically)
            "qwen3-4b-lora-sft": "/root/workspace/game_rl_training/Qwen3-4B-LoRA-SFT/final",
            # Merged LoRA model (full weights, works with vLLM) - latest
            "qwen3-4b-lora-merged": "/root/workspace/game_rl_training/Qwen3-4B-LoRA-SFT/merged_final",
        }
    
    def resolve_model_name(self, name: str) -> str:
        """Resolve alias to full model path/name"""
        return self.model_aliases.get(name, name)
    
    def _load_lora_model(self, model_name: str, resolved_name: str) -> dict:
        """Load a LoRA adapter using transformers + PEFT"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        logger.info(f"Detected LoRA adapter, loading with PEFT: {resolved_name}")
        
        # Get base model from adapter config
        base_model_name = get_lora_base_model(resolved_name)
        if not base_model_name:
            return {
                "status": "error",
                "model": model_name,
                "error": "Could not determine base model from adapter_config.json"
            }
        
        logger.info(f"Base model: {base_model_name}")
        start_time = time.time()
        
        try:
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, resolved_name)
            model.eval()
            
            # Load tokenizer from adapter directory
            tokenizer = AutoTokenizer.from_pretrained(resolved_name, trust_remote_code=True)
            
            load_time = time.time() - start_time
            
            self.engines[resolved_name] = {
                "engine": model,
                "tokenizer": tokenizer,
                "base_model": base_model_name,
                "loaded_at": datetime.now().isoformat(),
                "load_time": load_time,
                "alias": model_name if model_name != resolved_name else None,
                "backend": "peft",
            }
            self.current_model = resolved_name
            
            logger.info(f"LoRA model {model_name} loaded successfully with PEFT in {load_time:.2f}s")
            
            return {
                "status": "loaded",
                "model": model_name,
                "resolved_name": resolved_name,
                "base_model": base_model_name,
                "load_time": load_time,
                "backend": "peft"
            }
            
        except Exception as e:
            logger.error(f"Failed to load LoRA model {model_name}: {e}")
            return {
                "status": "error",
                "model": model_name,
                "error": str(e)
            }
    
    def _load_vllm_model(self, model_name: str, resolved_name: str) -> dict:
        """Load a full model using vLLM"""
        from vllm import LLM
        
        logger.info(f"Loading model with vLLM: {resolved_name}...")
        start_time = time.time()
        
        try:
            engine = LLM(
                model=resolved_name,
                trust_remote_code=True,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=4096,
                dtype="bfloat16",
            )
            
            load_time = time.time() - start_time
            
            self.engines[resolved_name] = {
                "engine": engine,
                "loaded_at": datetime.now().isoformat(),
                "load_time": load_time,
                "alias": model_name if model_name != resolved_name else None,
                "backend": "vllm",
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
    
    def load_model(self, model_name: str, force_reload: bool = False) -> dict:
        """Load a model - automatically detects LoRA adapters and uses appropriate backend"""
        import torch
        
        resolved_name = self.resolve_model_name(model_name)
        
        # Check if already loaded
        if resolved_name in self.engines and not force_reload:
            self.current_model = resolved_name
            backend = self.engines[resolved_name].get("backend", "vllm")
            logger.info(f"Model {model_name} already loaded ({backend}), switching to it")
            return {
                "status": "already_loaded",
                "model": model_name,
                "resolved_name": resolved_name,
                "backend": backend
            }
        
        # Unload existing models if at capacity
        if len(self.engines) >= self.max_models or force_reload:
            for existing_model in list(self.engines.keys()):
                self._unload_model_internal(existing_model)
            logger.info("Unloaded existing models to make room")
        
        # Check if this is a LoRA adapter
        if is_lora_adapter(resolved_name):
            return self._load_lora_model(model_name, resolved_name)
        else:
            return self._load_vllm_model(model_name, resolved_name)
    
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
        """Get a loaded engine (vLLM or PEFT model)"""
        resolved_name = self.resolve_model_name(model_name) if model_name else self.current_model
        
        if not resolved_name or resolved_name not in self.engines:
            return None
        
        return self.engines[resolved_name]["engine"]
    
    def get_backend(self, model_name: str = None) -> Optional[str]:
        """Get the backend type for a loaded model"""
        resolved_name = self.resolve_model_name(model_name) if model_name else self.current_model
        
        if not resolved_name or resolved_name not in self.engines:
            return None
        
        return self.engines[resolved_name].get("backend", "vllm")
    
    def generate(self, model_name: str, prompts: List[str], 
                 max_tokens: int = 512, temperature: float = 0.7, 
                 top_p: float = 0.9, stop: List[str] = None) -> List[str]:
        """Generate completions using vLLM or PEFT depending on backend"""
        import torch
        
        resolved_name = self.resolve_model_name(model_name) if model_name else self.current_model
        
        if resolved_name not in self.engines:
            raise ValueError(f"Model {model_name} not loaded")
        
        engine_info = self.engines[resolved_name]
        backend = engine_info.get("backend", "vllm")
        
        if backend == "vllm":
            # vLLM generation
            from vllm import SamplingParams
            
            engine = engine_info["engine"]
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.01,
                top_p=top_p,
                stop=stop,
            )
            
            outputs = engine.generate(prompts, sampling_params)
            return [output.outputs[0].text for output in outputs]
        
        else:
            # PEFT/transformers generation
            model = engine_info["engine"]
            tokenizer = engine_info["tokenizer"]
            
            results = []
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature if temperature > 0 else None,
                        top_p=top_p,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                
                # Decode only new tokens
                response = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                results.append(response)
            
            return results
    
    def list_models(self) -> dict:
        """List all loaded models"""
        models_info = []
        for name, info in self.engines.items():
            model_info = {
                "id": info.get("alias") or name,
                "resolved_name": name,
                "loaded_at": info["loaded_at"],
                "load_time": info.get("load_time"),
                "backend": info.get("backend", "vllm"),
                "is_current": name == self.current_model
            }
            # Add base model info for LoRA adapters
            if info.get("base_model"):
                model_info["base_model"] = info["base_model"]
            models_info.append(model_info)
        
        return {
            "object": "list",
            "data": models_info,
            "current_model": self.current_model,
            "available_aliases": list(self.model_aliases.keys())
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
            # For PEFT models, tokenizer is already loaded
            if resolved in manager.engines and manager.engines[resolved].get("backend") == "peft":
                tokenizer_cache[resolved] = manager.engines[resolved]["tokenizer"]
            else:
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
                "backend": manager.get_backend(model_name) or "vllm"
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
