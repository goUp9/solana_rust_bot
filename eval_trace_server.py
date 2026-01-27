#!/usr/bin/env python3
"""
Trace Environment Evaluation Server

This server evaluates models on trace tasks and provides a REST API for:
1. Running evaluations on trace tasks
2. Storing evaluation results (input, output, task_id, score, etc.)
3. Providing endpoints for an external dashboard to fetch results

Usage:
    # Start the evaluation server
    python eval_trace_server.py --port 8001
    
    # Preload a model for evaluation
    python eval_trace_server.py --port 8001 --preload qwen3-4b-lora-merged
    
    # Run with specific number of evaluation workers
    python eval_trace_server.py --port 8001 --workers 4

API Endpoints:
    # Evaluation
    POST /v1/eval/run          - Run evaluation on single task
    POST /v1/eval/batch        - Run batch evaluation on multiple tasks
    POST /v1/eval/benchmark    - Run full benchmark (N random tasks)
    
    # Real-time Status (for dashboard)
    GET  /v1/eval/active       - Get all active/running evaluations
    GET  /v1/eval/status/<id>  - Get status of specific evaluation/benchmark
    POST /v1/eval/cancel/<id>  - Cancel a running evaluation
    
    # Results
    GET  /v1/results           - Get all evaluation results (paginated)
    GET  /v1/results/<id>      - Get single result by ID
    GET  /v1/results/summary   - Get summary statistics
    GET  /v1/results/export    - Export results as JSON
    DELETE /v1/results         - Clear all results
    
    # Models
    POST /v1/models/load       - Load a model
    GET  /v1/models            - List loaded models
    GET  /v1/models/status     - Get model status
    
    # System
    GET  /health               - Health check
    GET  /v1/status            - Full system status
"""

import argparse
import asyncio
import gc
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, field
from threading import Lock, Thread
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from pathlib import Path

# Fix for Triton compilation errors
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# Add trace environment to path
sys.path.insert(0, '/root/workspace/affinetes/environments/trace')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class EvalResult:
    """Single evaluation result"""
    id: str
    task_id: Optional[int]
    dataset_index: int
    model_name: str
    
    # Input/Output
    prompt: str
    transformed_code: str
    stdin_input: str
    ground_truth: str
    model_output: str
    cleaned_output: str
    
    # Scores
    score: float  # 0.0 or 1.0
    test_result: str  # "0/1" or "1/1"
    
    # Metadata
    timestamp: str
    inference_time_ms: float
    input_tokens: int
    output_tokens: int
    seed: int
    
    # Optional extras
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class BenchmarkRun:
    """A complete benchmark run"""
    id: str
    model_name: str
    num_tasks: int
    completed_tasks: int
    correct_tasks: int
    accuracy: float
    avg_inference_time_ms: float
    total_time_seconds: float
    started_at: str
    completed_at: Optional[str]
    status: str  # "pending", "running", "completed", "failed", "cancelled"
    task_ids: List[int] = field(default_factory=list)
    result_ids: List[str] = field(default_factory=list)
    current_task_id: Optional[int] = None  # Currently evaluating task
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActiveEvaluation:
    """Tracks an active/running evaluation"""
    id: str
    eval_type: str  # "single", "batch", "benchmark"
    model_name: str
    started_at: str
    status: str  # "pending", "running", "completed", "failed", "cancelled"
    
    # Progress tracking
    total_tasks: int
    completed_tasks: int
    correct_tasks: int
    current_task_id: Optional[int] = None
    current_task_index: int = 0
    
    # Concurrency tracking
    concurrency: int = 1  # Number of parallel workers
    in_progress_tasks: List[int] = field(default_factory=list)  # Currently running task IDs
    
    # Timing
    avg_inference_time_ms: float = 0.0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    tasks_per_second: float = 0.0  # Throughput metric
    
    # Results
    result_ids: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Cancellation flag
    cancel_requested: bool = False
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["accuracy"] = self.correct_tasks / self.completed_tasks if self.completed_tasks > 0 else 0.0
        d["progress_percent"] = (self.completed_tasks / self.total_tasks * 100) if self.total_tasks > 0 else 0.0
        d["in_progress_count"] = len(self.in_progress_tasks)
        return d


# =============================================================================
# RESULTS STORAGE (SQLite for persistence)
# =============================================================================

class ResultsDB:
    """SQLite-based storage for evaluation results"""
    
    def __init__(self, db_path: str = "eval_results.db"):
        self.db_path = db_path
        self._lock = Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eval_results (
                    id TEXT PRIMARY KEY,
                    task_id INTEGER,
                    dataset_index INTEGER,
                    model_name TEXT,
                    prompt TEXT,
                    transformed_code TEXT,
                    stdin_input TEXT,
                    ground_truth TEXT,
                    model_output TEXT,
                    cleaned_output TEXT,
                    score REAL,
                    test_result TEXT,
                    timestamp TEXT,
                    inference_time_ms REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    seed INTEGER,
                    error TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id TEXT PRIMARY KEY,
                    model_name TEXT,
                    num_tasks INTEGER,
                    completed_tasks INTEGER,
                    correct_tasks INTEGER,
                    accuracy REAL,
                    avg_inference_time_ms REAL,
                    total_time_seconds REAL,
                    started_at TEXT,
                    completed_at TEXT,
                    status TEXT,
                    task_ids TEXT,
                    result_ids TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON eval_results(model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON eval_results(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON eval_results(score)")
            conn.commit()
    
    def save_result(self, result: EvalResult):
        """Save a single evaluation result"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO eval_results VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    result.id, result.task_id, result.dataset_index, result.model_name,
                    result.prompt, result.transformed_code, result.stdin_input,
                    result.ground_truth, result.model_output, result.cleaned_output,
                    result.score, result.test_result, result.timestamp,
                    result.inference_time_ms, result.input_tokens, result.output_tokens,
                    result.seed, result.error
                ))
                conn.commit()
    
    def get_result(self, result_id: str) -> Optional[Dict]:
        """Get a single result by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM eval_results WHERE id = ?", (result_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_results(
        self, 
        model_name: Optional[str] = None,
        limit: int = 100, 
        offset: int = 0,
        order_by: str = "timestamp",
        order_dir: str = "DESC"
    ) -> List[Dict]:
        """Get paginated results"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM eval_results"
            params = []
            
            if model_name:
                query += " WHERE model_name = ?"
                params.append(model_name)
            
            # Validate order_by to prevent SQL injection
            valid_columns = ["timestamp", "score", "inference_time_ms", "model_name", "task_id"]
            if order_by not in valid_columns:
                order_by = "timestamp"
            order_dir = "DESC" if order_dir.upper() == "DESC" else "ASC"
            
            query += f" ORDER BY {order_by} {order_dir} LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_summary(self, model_name: Optional[str] = None) -> Dict:
        """Get summary statistics"""
        with sqlite3.connect(self.db_path) as conn:
            where_clause = "WHERE model_name = ?" if model_name else ""
            params = [model_name] if model_name else []
            
            cursor = conn.execute(f"""
                SELECT 
                    COUNT(*) as total_evals,
                    SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END) as correct,
                    AVG(score) as accuracy,
                    AVG(inference_time_ms) as avg_inference_time_ms,
                    MIN(timestamp) as first_eval,
                    MAX(timestamp) as last_eval
                FROM eval_results {where_clause}
            """, params)
            row = cursor.fetchone()
            
            # Get per-model breakdown
            cursor = conn.execute("""
                SELECT 
                    model_name,
                    COUNT(*) as total,
                    SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END) as correct,
                    AVG(score) as accuracy
                FROM eval_results
                GROUP BY model_name
            """)
            by_model = [{"model": r[0], "total": r[1], "correct": r[2], "accuracy": r[3]} 
                        for r in cursor.fetchall()]
            
            return {
                "total_evaluations": row[0] or 0,
                "correct": row[1] or 0,
                "accuracy": row[2] or 0.0,
                "avg_inference_time_ms": row[3] or 0.0,
                "first_eval": row[4],
                "last_eval": row[5],
                "by_model": by_model
            }
    
    def get_count(self, model_name: Optional[str] = None) -> int:
        """Get total count of results"""
        with sqlite3.connect(self.db_path) as conn:
            if model_name:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM eval_results WHERE model_name = ?", 
                    (model_name,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM eval_results")
            return cursor.fetchone()[0]
    
    def clear_results(self, model_name: Optional[str] = None):
        """Clear all results or results for a specific model"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                if model_name:
                    conn.execute("DELETE FROM eval_results WHERE model_name = ?", (model_name,))
                else:
                    conn.execute("DELETE FROM eval_results")
                conn.commit()
    
    def save_benchmark(self, benchmark: BenchmarkRun):
        """Save a benchmark run"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO benchmark_runs VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    benchmark.id, benchmark.model_name, benchmark.num_tasks,
                    benchmark.completed_tasks, benchmark.correct_tasks,
                    benchmark.accuracy, benchmark.avg_inference_time_ms,
                    benchmark.total_time_seconds, benchmark.started_at,
                    benchmark.completed_at, benchmark.status,
                    json.dumps(benchmark.task_ids), json.dumps(benchmark.result_ids)
                ))
                conn.commit()
    
    def get_benchmarks(self, limit: int = 20) -> List[Dict]:
        """Get recent benchmark runs"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM benchmark_runs ORDER BY started_at DESC LIMIT ?",
                (limit,)
            )
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d["task_ids"] = json.loads(d["task_ids"]) if d["task_ids"] else []
                d["result_ids"] = json.loads(d["result_ids"]) if d["result_ids"] else []
                results.append(d)
            return results
    
    def export_results(self, model_name: Optional[str] = None) -> List[Dict]:
        """Export all results as JSON-serializable dicts"""
        return self.get_results(model_name=model_name, limit=100000, offset=0)


# =============================================================================
# MODEL MANAGER (reuse from serve_sft_model.py)
# =============================================================================

def is_lora_adapter(path: str) -> bool:
    """Check if a path contains a LoRA adapter"""
    if not os.path.isdir(path):
        return False
    has_adapter_config = os.path.exists(os.path.join(path, "adapter_config.json"))
    has_model_config = os.path.exists(os.path.join(path, "config.json"))
    return has_adapter_config and not has_model_config


def get_lora_base_model(adapter_path: str) -> Optional[str]:
    """Get the base model name from a LoRA adapter config"""
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
            return config.get("base_model_name_or_path")
    return None


class ModelManager:
    """Manages models for evaluation"""
    
    def __init__(self, gpu_memory_utilization: float = 0.85):
        self.engines: Dict[str, Any] = {}
        self.gpu_memory_utilization = gpu_memory_utilization
        self.current_model = None
        self.tokenizers: Dict[str, Any] = {}
        
        # Predefined model aliases
        self.model_aliases = {
            "qwen3-4b-sft": "/root/workspace/game_rl_training/Qwen3-4B-Instruct-2507-SFT/checkpoint-5928",
            "qwen3-0.6b-sft": "/root/workspace/game_rl_training/Qwen3-0.6B-SFT/checkpoint-5928",
            "qwen3-4b-lora-sft": "/root/workspace/game_rl_training/Qwen3-4B-LoRA-SFT/final",
            "qwen3-4b-lora-merged": "/root/workspace/game_rl_training/Qwen3-4B-LoRA-SFT/merged_final",
            "qwen3-4b-base": "Qwen/Qwen3-4B",
            "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct",
        }
    
    def resolve_model_name(self, name: str) -> str:
        """Resolve alias to full model path/name"""
        return self.model_aliases.get(name, name)
    
    def load_model(self, model_name: str, force_reload: bool = False) -> dict:
        """Load a model for inference"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        resolved_name = self.resolve_model_name(model_name)
        
        # Check if already loaded
        if resolved_name in self.engines and not force_reload:
            self.current_model = resolved_name
            logger.info(f"Model {model_name} already loaded")
            return {"status": "already_loaded", "model": model_name, "resolved_name": resolved_name}
        
        # Unload existing models
        for existing in list(self.engines.keys()):
            self._unload_internal(existing)
        
        logger.info(f"Loading model: {resolved_name}...")
        start_time = time.time()
        
        try:
            # Check if LoRA adapter
            if is_lora_adapter(resolved_name):
                from peft import PeftModel
                
                base_model_name = get_lora_base_model(resolved_name)
                if not base_model_name:
                    return {"status": "error", "error": "Could not determine base model"}
                
                logger.info(f"Loading LoRA adapter with base: {base_model_name}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                model = PeftModel.from_pretrained(base_model, resolved_name)
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained(resolved_name, trust_remote_code=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    resolved_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained(resolved_name, trust_remote_code=True)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            load_time = time.time() - start_time
            
            self.engines[resolved_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "loaded_at": datetime.now().isoformat(),
                "load_time": load_time,
                "alias": model_name if model_name != resolved_name else None,
            }
            self.tokenizers[resolved_name] = tokenizer
            self.current_model = resolved_name
            
            logger.info(f"Model loaded in {load_time:.2f}s")
            return {
                "status": "loaded",
                "model": model_name,
                "resolved_name": resolved_name,
                "load_time": load_time
            }
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {"status": "error", "error": str(e)}
    
    def _unload_internal(self, model_name: str):
        """Unload a model"""
        import torch
        
        if model_name in self.engines:
            del self.engines[model_name]["model"]
            del self.engines[model_name]
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
            
            if self.current_model == model_name:
                self.current_model = next(iter(self.engines.keys()), None)
            
            gc.collect()
            torch.cuda.empty_cache()
    
    def unload_model(self, model_name: str) -> dict:
        """Unload a model"""
        resolved = self.resolve_model_name(model_name)
        if resolved not in self.engines:
            return {"status": "not_loaded", "model": model_name}
        
        self._unload_internal(resolved)
        logger.info(f"Model {model_name} unloaded")
        return {"status": "unloaded", "model": model_name}
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate a response for chat messages"""
        import torch
        
        resolved = self.resolve_model_name(model_name) if model_name else self.current_model
        
        if resolved not in self.engines:
            raise ValueError(f"Model not loaded: {model_name}")
        
        model = self.engines[resolved]["model"]
        tokenizer = self.engines[resolved]["tokenizer"]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_tokens = inputs["input_ids"].shape[1]
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Decode only new tokens
        response_ids = outputs[0][input_tokens:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        output_tokens = len(response_ids)
        
        return {
            "response": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "inference_time_ms": inference_time,
        }
    
    def list_models(self) -> dict:
        """List loaded models"""
        models_info = []
        for name, info in self.engines.items():
            models_info.append({
                "id": info.get("alias") or name,
                "resolved_name": name,
                "loaded_at": info["loaded_at"],
                "load_time": info.get("load_time"),
                "is_current": name == self.current_model
            })
        
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
            "loaded_models": len(self.engines),
            "current_model": self.current_model,
            "models": self.list_models()["data"],
            "gpu_memory": gpu_memory,
            "available_aliases": self.model_aliases
        }


# =============================================================================
# ACTIVE EVALUATION MANAGER
# =============================================================================

class ActiveEvaluationManager:
    """Manages and tracks all active/running evaluations"""
    
    def __init__(self):
        self._active: Dict[str, ActiveEvaluation] = {}
        self._lock = Lock()
        self._history: List[ActiveEvaluation] = []  # Keep recent completed
        self._max_history = 50
    
    def create(
        self, 
        eval_type: str, 
        model_name: str, 
        total_tasks: int,
        task_ids: Optional[List[int]] = None,
        concurrency: int = 1,
    ) -> ActiveEvaluation:
        """Create a new active evaluation"""
        eval_id = str(uuid.uuid4())
        evaluation = ActiveEvaluation(
            id=eval_id,
            eval_type=eval_type,
            model_name=model_name,
            started_at=datetime.now().isoformat(),
            status="pending",
            total_tasks=total_tasks,
            completed_tasks=0,
            correct_tasks=0,
            concurrency=concurrency,
        )
        
        with self._lock:
            self._active[eval_id] = evaluation
        
        logger.info(f"Created {eval_type} evaluation {eval_id} for {model_name} ({total_tasks} tasks, concurrency={concurrency})")
        return evaluation
    
    def update(
        self,
        eval_id: str,
        status: Optional[str] = None,
        completed_tasks: Optional[int] = None,
        correct_tasks: Optional[int] = None,
        current_task_id: Optional[int] = None,
        current_task_index: Optional[int] = None,
        avg_inference_time_ms: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
        result_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[ActiveEvaluation]:
        """Update an active evaluation"""
        with self._lock:
            if eval_id not in self._active:
                return None
            
            evaluation = self._active[eval_id]
            
            if status is not None:
                evaluation.status = status
            if completed_tasks is not None:
                evaluation.completed_tasks = completed_tasks
            if correct_tasks is not None:
                evaluation.correct_tasks = correct_tasks
            if current_task_id is not None:
                evaluation.current_task_id = current_task_id
            if current_task_index is not None:
                evaluation.current_task_index = current_task_index
            if avg_inference_time_ms is not None:
                evaluation.avg_inference_time_ms = avg_inference_time_ms
            if elapsed_seconds is not None:
                evaluation.elapsed_seconds = elapsed_seconds
                # Estimate remaining time
                if evaluation.completed_tasks > 0:
                    avg_time_per_task = elapsed_seconds / evaluation.completed_tasks
                    remaining_tasks = evaluation.total_tasks - evaluation.completed_tasks
                    evaluation.estimated_remaining_seconds = avg_time_per_task * remaining_tasks
            if result_id is not None:
                evaluation.result_ids.append(result_id)
            if error_message is not None:
                evaluation.error_message = error_message
            
            return evaluation
    
    def complete(self, eval_id: str, status: str = "completed") -> Optional[ActiveEvaluation]:
        """Mark an evaluation as complete and move to history"""
        with self._lock:
            if eval_id not in self._active:
                return None
            
            evaluation = self._active[eval_id]
            evaluation.status = status
            evaluation.completed_at = datetime.now().isoformat()
            
            # Move to history
            self._history.insert(0, evaluation)
            if len(self._history) > self._max_history:
                self._history = self._history[:self._max_history]
            
            del self._active[eval_id]
            
            logger.info(f"Evaluation {eval_id} {status}")
            return evaluation
    
    def cancel(self, eval_id: str) -> bool:
        """Request cancellation of an evaluation"""
        with self._lock:
            if eval_id not in self._active:
                return False
            
            self._active[eval_id].cancel_requested = True
            logger.info(f"Cancellation requested for evaluation {eval_id}")
            return True
    
    def is_cancelled(self, eval_id: str) -> bool:
        """Check if evaluation has been cancelled"""
        with self._lock:
            if eval_id not in self._active:
                return False
            return self._active[eval_id].cancel_requested
    
    def add_in_progress_task(self, eval_id: str, task_id: int):
        """Mark a task as in-progress"""
        with self._lock:
            if eval_id in self._active:
                if task_id not in self._active[eval_id].in_progress_tasks:
                    self._active[eval_id].in_progress_tasks.append(task_id)
    
    def remove_in_progress_task(self, eval_id: str, task_id: int):
        """Remove a task from in-progress list"""
        with self._lock:
            if eval_id in self._active:
                if task_id in self._active[eval_id].in_progress_tasks:
                    self._active[eval_id].in_progress_tasks.remove(task_id)
    
    def increment_completed(
        self, 
        eval_id: str, 
        is_correct: bool,
        inference_time_ms: float,
        result_id: str,
        elapsed_seconds: float,
    ):
        """Thread-safe increment of completed tasks"""
        with self._lock:
            if eval_id not in self._active:
                return
            
            evaluation = self._active[eval_id]
            evaluation.completed_tasks += 1
            if is_correct:
                evaluation.correct_tasks += 1
            evaluation.result_ids.append(result_id)
            
            # Update timing metrics
            evaluation.elapsed_seconds = elapsed_seconds
            
            # Calculate average inference time (rolling average)
            n = evaluation.completed_tasks
            if n == 1:
                evaluation.avg_inference_time_ms = inference_time_ms
            else:
                evaluation.avg_inference_time_ms = (
                    (evaluation.avg_inference_time_ms * (n - 1) + inference_time_ms) / n
                )
            
            # Calculate throughput
            if elapsed_seconds > 0:
                evaluation.tasks_per_second = evaluation.completed_tasks / elapsed_seconds
            
            # Estimate remaining time based on throughput
            if evaluation.tasks_per_second > 0:
                remaining_tasks = evaluation.total_tasks - evaluation.completed_tasks
                evaluation.estimated_remaining_seconds = remaining_tasks / evaluation.tasks_per_second
    
    def get(self, eval_id: str) -> Optional[ActiveEvaluation]:
        """Get an evaluation by ID (active or from history)"""
        with self._lock:
            if eval_id in self._active:
                return self._active[eval_id]
            
            # Check history
            for evaluation in self._history:
                if evaluation.id == eval_id:
                    return evaluation
            
            return None
    
    def get_active(self) -> List[ActiveEvaluation]:
        """Get all active evaluations"""
        with self._lock:
            return list(self._active.values())
    
    def get_all(self, include_history: bool = True) -> Dict[str, List[Dict]]:
        """Get all evaluations (active and recent history)"""
        with self._lock:
            result = {
                "active": [e.to_dict() for e in self._active.values()],
                "active_count": len(self._active),
            }
            if include_history:
                result["recent_completed"] = [e.to_dict() for e in self._history[:10]]
            return result


# =============================================================================
# TRACE EVALUATOR
# =============================================================================

class TraceEvaluator:
    """Evaluates models on trace tasks"""
    
    # Maximum allowed concurrency (GPU memory constraint)
    MAX_CONCURRENCY = 8
    
    def __init__(
        self, 
        model_manager: ModelManager, 
        results_db: ResultsDB,
        active_manager: Optional[ActiveEvaluationManager] = None,
        default_concurrency: int = 1,
    ):
        self.model_manager = model_manager
        self.results_db = results_db
        self.active_manager = active_manager or ActiveEvaluationManager()
        self.default_concurrency = min(default_concurrency, self.MAX_CONCURRENCY)
        self._trace_task = None
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=16)  # For concurrent evaluations
    
    @property
    def trace_task(self):
        """Lazy load TraceTask"""
        if self._trace_task is None:
            from trace_task import TraceTask
            self._trace_task = TraceTask()
        return self._trace_task
    
    async def evaluate_single(
        self,
        task_id: Optional[int] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> EvalResult:
        """Evaluate a single trace task"""
        from trace_task import clean_llm_prediction, compare_outputs
        
        # Generate challenge
        challenge = await self.trace_task.generate(task_id=task_id)
        
        # Prepare messages
        messages = [{"role": "user", "content": challenge.prompt}]
        
        # Generate response
        try:
            gen_result = self.model_manager.generate(
                messages=messages,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            model_output = gen_result["response"]
            error = None
        except Exception as e:
            model_output = ""
            gen_result = {"input_tokens": 0, "output_tokens": 0, "inference_time_ms": 0}
            error = str(e)
        
        # Evaluate
        cleaned_output = clean_llm_prediction(model_output)
        ground_truth = challenge.extra.get("ground_truth", "")
        score = 1.0 if compare_outputs(ground_truth, cleaned_output) else 0.0
        test_result = "1/1" if score > 0 else "0/1"
        
        # Create result
        result = EvalResult(
            id=str(uuid.uuid4()),
            task_id=task_id,
            dataset_index=challenge.extra.get("dataset_index", -1),
            model_name=model_name or self.model_manager.current_model or "unknown",
            prompt=challenge.prompt,
            transformed_code=challenge.extra.get("transformed_code", ""),
            stdin_input=challenge.extra.get("inputs", ""),
            ground_truth=ground_truth,
            model_output=model_output,
            cleaned_output=cleaned_output,
            score=score,
            test_result=test_result,
            timestamp=datetime.now().isoformat(),
            inference_time_ms=gen_result["inference_time_ms"],
            input_tokens=gen_result["input_tokens"],
            output_tokens=gen_result["output_tokens"],
            seed=challenge.extra.get("seed", 0),
            error=error,
        )
        
        # Save to database
        self.results_db.save_result(result)
        
        return result
    
    async def evaluate_batch(
        self,
        task_ids: List[int],
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> List[EvalResult]:
        """Evaluate multiple tasks"""
        results = []
        for task_id in task_ids:
            result = await self.evaluate_single(
                task_id=task_id,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            results.append(result)
        return results
    
    async def run_benchmark(
        self,
        num_tasks: int = 100,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        random_seed: int = 42,
        progress_callback=None,
        eval_id: Optional[str] = None,  # For tracking
        concurrency: int = 1,  # Number of parallel workers
    ) -> BenchmarkRun:
        """Run a full benchmark on random tasks with optional concurrency"""
        import random
        random.seed(random_seed)
        
        # Validate and cap concurrency
        concurrency = max(1, min(concurrency, self.MAX_CONCURRENCY))
        
        resolved_model = model_name or self.model_manager.current_model or "unknown"
        
        # Generate random task IDs
        dataset_size = len(self.trace_task.dataset)
        task_ids = random.sample(range(dataset_size), min(num_tasks, dataset_size))
        
        benchmark = BenchmarkRun(
            id=eval_id or str(uuid.uuid4()),
            model_name=resolved_model,
            num_tasks=len(task_ids),
            completed_tasks=0,
            correct_tasks=0,
            accuracy=0.0,
            avg_inference_time_ms=0.0,
            total_time_seconds=0.0,
            started_at=datetime.now().isoformat(),
            completed_at=None,
            status="running",
            task_ids=task_ids,
            result_ids=[],
        )
        
        self.results_db.save_benchmark(benchmark)
        
        # Update active evaluation status
        if eval_id and self.active_manager:
            self.active_manager.update(eval_id, status="running")
        
        start_time = time.time()
        
        if concurrency == 1:
            # Sequential execution (original behavior)
            await self._run_benchmark_sequential(
                benchmark, task_ids, model_name, temperature, max_tokens,
                eval_id, start_time, progress_callback
            )
        else:
            # Concurrent execution
            await self._run_benchmark_concurrent(
                benchmark, task_ids, model_name, temperature, max_tokens,
                eval_id, start_time, progress_callback, concurrency
            )
        
        return benchmark
    
    async def _run_benchmark_sequential(
        self,
        benchmark: BenchmarkRun,
        task_ids: List[int],
        model_name: Optional[str],
        temperature: float,
        max_tokens: int,
        eval_id: Optional[str],
        start_time: float,
        progress_callback,
    ):
        """Run benchmark tasks sequentially"""
        total_inference_time = 0.0
        
        for i, task_id in enumerate(task_ids):
            # Check for cancellation
            if eval_id and self.active_manager and self.active_manager.is_cancelled(eval_id):
                logger.info(f"Benchmark {eval_id} cancelled at task {i+1}/{len(task_ids)}")
                benchmark.status = "cancelled"
                benchmark.completed_at = datetime.now().isoformat()
                benchmark.total_time_seconds = time.time() - start_time
                self.results_db.save_benchmark(benchmark)
                
                if self.active_manager:
                    self.active_manager.complete(eval_id, status="cancelled")
                return
            
            # Update current task in tracking
            benchmark.current_task_id = task_id
            if eval_id and self.active_manager:
                self.active_manager.update(
                    eval_id,
                    current_task_id=task_id,
                    current_task_index=i,
                    elapsed_seconds=time.time() - start_time,
                )
            
            try:
                result = await self.evaluate_single(
                    task_id=task_id,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                benchmark.completed_tasks += 1
                benchmark.result_ids.append(result.id)
                
                if result.score > 0:
                    benchmark.correct_tasks += 1
                
                total_inference_time += result.inference_time_ms
                
                benchmark.accuracy = benchmark.correct_tasks / benchmark.completed_tasks
                benchmark.avg_inference_time_ms = total_inference_time / benchmark.completed_tasks
                benchmark.total_time_seconds = time.time() - start_time
                
                # Update benchmark in DB
                self.results_db.save_benchmark(benchmark)
                
                # Update active evaluation
                if eval_id and self.active_manager:
                    self.active_manager.update(
                        eval_id,
                        completed_tasks=benchmark.completed_tasks,
                        correct_tasks=benchmark.correct_tasks,
                        avg_inference_time_ms=benchmark.avg_inference_time_ms,
                        elapsed_seconds=benchmark.total_time_seconds,
                        result_id=result.id,
                    )
                
                if progress_callback:
                    progress_callback(benchmark)
                
                logger.info(
                    f"Benchmark [{i+1}/{len(task_ids)}] "
                    f"Task {task_id}: {result.test_result} "
                    f"(Accuracy: {benchmark.accuracy:.1%})"
                )
                
            except Exception as e:
                logger.error(f"Error evaluating task {task_id}: {e}")
                benchmark.completed_tasks += 1
                if eval_id and self.active_manager:
                    self.active_manager.update(
                        eval_id,
                        completed_tasks=benchmark.completed_tasks,
                        elapsed_seconds=time.time() - start_time,
                    )
        
        benchmark.completed_at = datetime.now().isoformat()
        benchmark.status = "completed"
        benchmark.current_task_id = None
        benchmark.total_time_seconds = time.time() - start_time
        self.results_db.save_benchmark(benchmark)
        
        # Mark evaluation as complete
        if eval_id and self.active_manager:
            self.active_manager.complete(eval_id, status="completed")
    
    async def _run_benchmark_concurrent(
        self,
        benchmark: BenchmarkRun,
        task_ids: List[int],
        model_name: Optional[str],
        temperature: float,
        max_tokens: int,
        eval_id: Optional[str],
        start_time: float,
        progress_callback,
        concurrency: int,
    ):
        """Run benchmark tasks concurrently using asyncio semaphore"""
        import asyncio
        
        semaphore = asyncio.Semaphore(concurrency)
        results_lock = asyncio.Lock()
        total_inference_time = 0.0
        
        async def evaluate_task(task_id: int, task_index: int):
            nonlocal total_inference_time
            
            # Check for cancellation before starting
            if eval_id and self.active_manager and self.active_manager.is_cancelled(eval_id):
                return None
            
            async with semaphore:
                # Check again after acquiring semaphore
                if eval_id and self.active_manager and self.active_manager.is_cancelled(eval_id):
                    return None
                
                # Track in-progress task
                if eval_id and self.active_manager:
                    self.active_manager.add_in_progress_task(eval_id, task_id)
                
                try:
                    result = await self.evaluate_single(
                        task_id=task_id,
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    
                    # Thread-safe update of results
                    async with results_lock:
                        benchmark.completed_tasks += 1
                        benchmark.result_ids.append(result.id)
                        
                        if result.score > 0:
                            benchmark.correct_tasks += 1
                        
                        total_inference_time += result.inference_time_ms
                        
                        benchmark.accuracy = benchmark.correct_tasks / benchmark.completed_tasks
                        benchmark.avg_inference_time_ms = total_inference_time / benchmark.completed_tasks
                        benchmark.total_time_seconds = time.time() - start_time
                        
                        # Update benchmark in DB periodically (every 5 completions)
                        if benchmark.completed_tasks % 5 == 0 or benchmark.completed_tasks == len(task_ids):
                            self.results_db.save_benchmark(benchmark)
                    
                    # Update active evaluation tracking
                    if eval_id and self.active_manager:
                        self.active_manager.increment_completed(
                            eval_id,
                            is_correct=result.score > 0,
                            inference_time_ms=result.inference_time_ms,
                            result_id=result.id,
                            elapsed_seconds=time.time() - start_time,
                        )
                    
                    if progress_callback:
                        progress_callback(benchmark)
                    
                    completed = benchmark.completed_tasks
                    total = len(task_ids)
                    logger.info(
                        f"Benchmark [{completed}/{total}] "
                        f"Task {task_id}: {result.test_result} "
                        f"(Accuracy: {benchmark.accuracy:.1%}, "
                        f"Rate: {completed/(time.time()-start_time):.1f}/s)"
                    )
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error evaluating task {task_id}: {e}")
                    async with results_lock:
                        benchmark.completed_tasks += 1
                    if eval_id and self.active_manager:
                        self.active_manager.update(
                            eval_id,
                            completed_tasks=benchmark.completed_tasks,
                            elapsed_seconds=time.time() - start_time,
                        )
                    return None
                    
                finally:
                    # Remove from in-progress
                    if eval_id and self.active_manager:
                        self.active_manager.remove_in_progress_task(eval_id, task_id)
        
        # Create tasks for all evaluations
        tasks = [
            evaluate_task(task_id, i) 
            for i, task_id in enumerate(task_ids)
        ]
        
        # Run all tasks concurrently (semaphore controls actual parallelism)
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check if cancelled
        if eval_id and self.active_manager and self.active_manager.is_cancelled(eval_id):
            benchmark.status = "cancelled"
            benchmark.completed_at = datetime.now().isoformat()
            benchmark.total_time_seconds = time.time() - start_time
            self.results_db.save_benchmark(benchmark)
            self.active_manager.complete(eval_id, status="cancelled")
            return
        
        # Finalize benchmark
        benchmark.completed_at = datetime.now().isoformat()
        benchmark.status = "completed"
        benchmark.current_task_id = None
        benchmark.total_time_seconds = time.time() - start_time
        self.results_db.save_benchmark(benchmark)
        
        # Mark evaluation as complete
        if eval_id and self.active_manager:
            self.active_manager.complete(eval_id, status="completed")
        
        logger.info(
            f"Benchmark completed: {benchmark.correct_tasks}/{benchmark.completed_tasks} correct "
            f"({benchmark.accuracy:.1%}) in {benchmark.total_time_seconds:.1f}s "
            f"(concurrency={concurrency})"
        )
    
    def run_benchmark_async(
        self,
        num_tasks: int = 100,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        random_seed: int = 42,
        concurrency: int = 1,
    ) -> ActiveEvaluation:
        """Start a benchmark asynchronously and return tracking info"""
        resolved_model = model_name or self.model_manager.current_model or "unknown"
        
        # Validate concurrency
        concurrency = max(1, min(concurrency, self.MAX_CONCURRENCY))
        
        # Create active evaluation entry
        evaluation = self.active_manager.create(
            eval_type="benchmark",
            model_name=resolved_model,
            total_tasks=num_tasks,
            concurrency=concurrency,
        )
        
        # Run benchmark in background thread
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.run_benchmark(
                    num_tasks=num_tasks,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    random_seed=random_seed,
                    eval_id=evaluation.id,
                    concurrency=concurrency,
                ))
            except Exception as e:
                logger.error(f"Benchmark {evaluation.id} failed: {e}")
                self.active_manager.update(evaluation.id, error_message=str(e))
                self.active_manager.complete(evaluation.id, status="failed")
            finally:
                loop.close()
        
        thread = Thread(target=run_in_thread, daemon=True)
        thread.start()
        
        return evaluation


# =============================================================================
# FLASK APP
# =============================================================================

def create_app(
    model_manager: ModelManager,
    results_db: ResultsDB,
    evaluator: TraceEvaluator,
    active_manager: ActiveEvaluationManager,
):
    """Create Flask app with evaluation endpoints"""
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
    except ImportError:
        print("❌ Required packages not installed. Install with:")
        print("   pip install flask flask-cors")
        sys.exit(1)
    
    app = Flask(__name__)
    CORS(app)  # Enable CORS for external dashboard access
    
    # Create async event loop for evaluator
    loop = asyncio.new_event_loop()
    
    def run_async(coro):
        """Run async function in the event loop"""
        return loop.run_until_complete(coro)
    
    # =========================================================================
    # EVALUATION ENDPOINTS
    # =========================================================================
    
    @app.route("/v1/eval/run", methods=["POST"])
    def eval_single():
        """Run evaluation on a single task"""
        data = request.json or {}
        task_id = data.get("task_id")  # None = random
        model_name = data.get("model")
        temperature = data.get("temperature", 0.0)
        max_tokens = data.get("max_tokens", 2048)
        
        if not model_manager.current_model:
            return jsonify({"error": "No model loaded"}), 400
        
        try:
            result = run_async(evaluator.evaluate_single(
                task_id=task_id,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            ))
            
            return jsonify({
                "status": "success",
                "result": result.to_dict()
            })
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/v1/eval/batch", methods=["POST"])
    def eval_batch():
        """Run evaluation on multiple tasks"""
        data = request.json or {}
        task_ids = data.get("task_ids", [])
        model_name = data.get("model")
        temperature = data.get("temperature", 0.0)
        max_tokens = data.get("max_tokens", 2048)
        
        if not task_ids:
            return jsonify({"error": "task_ids required"}), 400
        
        if not model_manager.current_model:
            return jsonify({"error": "No model loaded"}), 400
        
        try:
            results = run_async(evaluator.evaluate_batch(
                task_ids=task_ids,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            ))
            
            return jsonify({
                "status": "success",
                "results": [r.to_dict() for r in results],
                "summary": {
                    "total": len(results),
                    "correct": sum(1 for r in results if r.score > 0),
                    "accuracy": sum(r.score for r in results) / len(results) if results else 0
                }
            })
            
        except Exception as e:
            logger.error(f"Batch evaluation error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/v1/eval/benchmark", methods=["POST"])
    def run_benchmark():
        """Run a full benchmark (async by default, returns immediately with tracking ID)
        
        Request body:
            num_tasks: int - Number of tasks to evaluate (default: 100, max: 1000)
            model: str - Model name/alias (optional, uses current model)
            temperature: float - Sampling temperature (default: 0.0)
            max_tokens: int - Max output tokens (default: 2048)
            seed: int - Random seed for task selection (default: 42)
            sync: bool - If True, wait for completion (default: False)
            concurrency: int - Number of parallel evaluations (default: 1, max: 8)
        """
        data = request.json or {}
        num_tasks = data.get("num_tasks", 100)
        model_name = data.get("model")
        temperature = data.get("temperature", 0.0)
        max_tokens = data.get("max_tokens", 2048)
        random_seed = data.get("seed", 42)
        run_sync = data.get("sync", False)  # If True, wait for completion
        concurrency = data.get("concurrency", 1)  # Parallel workers
        
        if not model_manager.current_model:
            return jsonify({"error": "No model loaded"}), 400
        
        if num_tasks > 1000:
            return jsonify({"error": "num_tasks must be <= 1000"}), 400
        
        # Validate and cap concurrency
        concurrency = max(1, min(concurrency, evaluator.MAX_CONCURRENCY))
        
        try:
            if run_sync:
                # Synchronous execution (blocks until complete)
                benchmark = run_async(evaluator.run_benchmark(
                    num_tasks=num_tasks,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    random_seed=random_seed,
                    concurrency=concurrency,
                ))
                
                return jsonify({
                    "status": "completed",
                    "benchmark": benchmark.to_dict()
                })
            else:
                # Async execution (returns immediately)
                evaluation = evaluator.run_benchmark_async(
                    num_tasks=num_tasks,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    random_seed=random_seed,
                    concurrency=concurrency,
                )
                
                return jsonify({
                    "status": "started",
                    "message": f"Benchmark started in background (concurrency={concurrency})",
                    "eval_id": evaluation.id,
                    "evaluation": evaluation.to_dict(),
                    "endpoints": {
                        "status": f"/v1/eval/status/{evaluation.id}",
                        "cancel": f"/v1/eval/cancel/{evaluation.id}",
                        "active": "/v1/eval/active"
                    }
                })
            
        except Exception as e:
            logger.error(f"Benchmark error: {e}")
            return jsonify({"error": str(e)}), 500
    
    # =========================================================================
    # REAL-TIME STATUS ENDPOINTS (for dashboard)
    # =========================================================================
    
    @app.route("/v1/eval/active", methods=["GET"])
    def get_active_evaluations():
        """Get all active/running evaluations - for real-time dashboard"""
        include_history = request.args.get("include_history", "true").lower() == "true"
        result = active_manager.get_all(include_history=include_history)
        return jsonify(result)
    
    @app.route("/v1/eval/config", methods=["GET"])
    def get_eval_config():
        """Get evaluation configuration and limits"""
        return jsonify({
            "max_concurrency": evaluator.MAX_CONCURRENCY,
            "default_concurrency": evaluator.default_concurrency,
            "max_tasks_per_benchmark": 1000,
            "supported_parameters": {
                "num_tasks": {"type": "int", "default": 100, "max": 1000},
                "concurrency": {"type": "int", "default": 1, "max": evaluator.MAX_CONCURRENCY},
                "temperature": {"type": "float", "default": 0.0},
                "max_tokens": {"type": "int", "default": 2048},
                "seed": {"type": "int", "default": 42},
                "sync": {"type": "bool", "default": False},
            }
        })
    
    @app.route("/v1/eval/status/<eval_id>", methods=["GET"])
    def get_evaluation_status(eval_id):
        """Get status of a specific evaluation by ID"""
        evaluation = active_manager.get(eval_id)
        
        if evaluation is None:
            # Check if it's a benchmark ID in the database
            benchmarks = results_db.get_benchmarks(limit=100)
            for b in benchmarks:
                if b["id"] == eval_id:
                    return jsonify({
                        "found": True,
                        "source": "database",
                        "evaluation": b
                    })
            
            return jsonify({"error": "Evaluation not found"}), 404
        
        return jsonify({
            "found": True,
            "source": "active" if evaluation.status in ["pending", "running"] else "history",
            "evaluation": evaluation.to_dict()
        })
    
    @app.route("/v1/eval/cancel/<eval_id>", methods=["POST"])
    def cancel_evaluation(eval_id):
        """Cancel a running evaluation"""
        success = active_manager.cancel(eval_id)
        
        if not success:
            evaluation = active_manager.get(eval_id)
            if evaluation:
                return jsonify({
                    "error": f"Evaluation already {evaluation.status}",
                    "evaluation": evaluation.to_dict()
                }), 400
            return jsonify({"error": "Evaluation not found"}), 404
        
        return jsonify({
            "status": "cancellation_requested",
            "eval_id": eval_id,
            "message": "Cancellation requested. Evaluation will stop after current task."
        })
    
    # =========================================================================
    # RESULTS ENDPOINTS
    # =========================================================================
    
    @app.route("/v1/results", methods=["GET"])
    def get_results():
        """Get paginated evaluation results"""
        model_name = request.args.get("model")
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
        order_by = request.args.get("order_by", "timestamp")
        order_dir = request.args.get("order_dir", "DESC")
        
        results = results_db.get_results(
            model_name=model_name,
            limit=limit,
            offset=offset,
            order_by=order_by,
            order_dir=order_dir,
        )
        total = results_db.get_count(model_name=model_name)
        
        return jsonify({
            "object": "list",
            "data": results,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(results) < total
        })
    
    @app.route("/v1/results/<result_id>", methods=["GET"])
    def get_result(result_id):
        """Get a single result by ID"""
        result = results_db.get_result(result_id)
        if result is None:
            return jsonify({"error": "Result not found"}), 404
        return jsonify(result)
    
    @app.route("/v1/results/summary", methods=["GET"])
    def get_summary():
        """Get summary statistics"""
        model_name = request.args.get("model")
        summary = results_db.get_summary(model_name=model_name)
        return jsonify(summary)
    
    @app.route("/v1/results/export", methods=["GET"])
    def export_results():
        """Export all results as JSON"""
        model_name = request.args.get("model")
        results = results_db.export_results(model_name=model_name)
        return jsonify({
            "object": "export",
            "data": results,
            "total": len(results),
            "exported_at": datetime.now().isoformat()
        })
    
    @app.route("/v1/results", methods=["DELETE"])
    def clear_results():
        """Clear all results"""
        model_name = request.json.get("model") if request.json else None
        results_db.clear_results(model_name=model_name)
        return jsonify({"status": "cleared", "model": model_name})
    
    @app.route("/v1/benchmarks", methods=["GET"])
    def get_benchmarks():
        """Get recent benchmark runs"""
        limit = int(request.args.get("limit", 20))
        benchmarks = results_db.get_benchmarks(limit=limit)
        return jsonify({
            "object": "list",
            "data": benchmarks
        })
    
    # =========================================================================
    # MODEL ENDPOINTS
    # =========================================================================
    
    @app.route("/v1/models/load", methods=["POST"])
    def load_model():
        """Load a model"""
        data = request.json or {}
        model_name = data.get("model")
        force_reload = data.get("force_reload", False)
        
        if not model_name:
            return jsonify({"error": "model name required"}), 400
        
        result = model_manager.load_model(model_name, force_reload=force_reload)
        status_code = 200 if result["status"] != "error" else 500
        return jsonify(result), status_code
    
    @app.route("/v1/models/unload", methods=["POST"])
    def unload_model():
        """Unload a model"""
        data = request.json or {}
        model_name = data.get("model")
        
        if not model_name:
            return jsonify({"error": "model name required"}), 400
        
        result = model_manager.unload_model(model_name)
        return jsonify(result)
    
    @app.route("/v1/models", methods=["GET"])
    def list_models():
        """List loaded models"""
        return jsonify(model_manager.list_models())
    
    @app.route("/v1/models/status", methods=["GET"])
    def model_status():
        """Get detailed model status"""
        return jsonify(model_manager.get_status())
    
    # =========================================================================
    # SYSTEM ENDPOINTS
    # =========================================================================
    
    @app.route("/health", methods=["GET"])
    def health():
        """Health check"""
        active_evals = active_manager.get_active()
        return jsonify({
            "status": "healthy",
            "models_loaded": len(model_manager.engines),
            "current_model": model_manager.current_model,
            "total_evaluations": results_db.get_count(),
            "active_evaluations": len(active_evals),
            "active_eval_ids": [e.id for e in active_evals]
        })
    
    @app.route("/v1/status", methods=["GET"])
    def full_status():
        """Full system status including active evaluations"""
        return jsonify({
            "model_status": model_manager.get_status(),
            "results_summary": results_db.get_summary(),
            "recent_benchmarks": results_db.get_benchmarks(limit=5),
            "active_evaluations": active_manager.get_all(include_history=True)
        })
    
    return app


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Trace Environment Evaluation Server")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--preload", type=str, default=None, help="Model to preload on startup")
    parser.add_argument("--db", type=str, default="eval_results.db", help="SQLite database path")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="GPU memory utilization")
    parser.add_argument("--default-concurrency", type=int, default=1, 
                        help=f"Default concurrency for benchmarks (max: {TraceEvaluator.MAX_CONCURRENCY})")
    
    args = parser.parse_args()
    
    # Create components
    model_manager = ModelManager(gpu_memory_utilization=args.gpu_memory_utilization)
    results_db = ResultsDB(db_path=args.db)
    active_manager = ActiveEvaluationManager()
    evaluator = TraceEvaluator(
        model_manager, 
        results_db, 
        active_manager,
        default_concurrency=args.default_concurrency
    )
    
    # Preload model if specified
    if args.preload:
        logger.info(f"Preloading model: {args.preload}")
        result = model_manager.load_model(args.preload)
        if result["status"] == "error":
            logger.error(f"Failed to preload model: {result.get('error')}")
            sys.exit(1)
    
    # Create and run app
    app = create_app(model_manager, results_db, evaluator, active_manager)
    
    print(f"\n🔬 Trace Environment Evaluation Server")
    print("=" * 60)
    print(f"Host: {args.host}:{args.port}")
    print(f"Database: {args.db}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Max Concurrency: {evaluator.MAX_CONCURRENCY}")
    print(f"Default Concurrency: {evaluator.default_concurrency}")
    print("\nEvaluation Endpoints:")
    print(f"  POST /v1/eval/run         - Evaluate single task")
    print(f"  POST /v1/eval/batch       - Evaluate multiple tasks")
    print(f"  POST /v1/eval/benchmark   - Run benchmark (supports concurrency)")
    print(f"  GET  /v1/eval/config      - Get evaluation config/limits")
    print("\nReal-time Status Endpoints (for dashboard):")
    print(f"  GET  /v1/eval/active      - Get all active evaluations")
    print(f"  GET  /v1/eval/status/<id> - Get evaluation status by ID")
    print(f"  POST /v1/eval/cancel/<id> - Cancel running evaluation")
    print("\nResults Endpoints:")
    print(f"  GET  /v1/results          - Get results (paginated)")
    print(f"  GET  /v1/results/<id>     - Get single result")
    print(f"  GET  /v1/results/summary  - Get summary stats")
    print(f"  GET  /v1/results/export   - Export all results")
    print(f"  GET  /v1/benchmarks       - Get benchmark runs")
    print("\nModel Endpoints:")
    print(f"  POST /v1/models/load      - Load a model")
    print(f"  GET  /v1/models           - List loaded models")
    print(f"  GET  /v1/models/status    - Model status")
    print("\nAvailable model aliases:")
    for alias, path in model_manager.model_aliases.items():
        print(f"  {alias:25} -> {path}")
    print("=" * 60 + "\n")
    
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
