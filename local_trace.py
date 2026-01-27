#!/usr/bin/env python3
"""
Local Trace task runner for PPO training.

This module runs the Trace task (code tracing) locally without Docker,
allowing direct model inference for PPO training.

The Trace task:
1. Takes Python code with injected debug prints
2. Model predicts the exact stdout output
3. Binary reward: 1.0 if exact match, 0.0 otherwise
"""

import os
import sys
import ast
import random
import tempfile
import subprocess
import re
from typing import List, Tuple, Optional, Dict, Any, Generator
from dataclasses import dataclass

import torch
import numpy as np

# Try to import datasets for HuggingFace dataset loading
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets not available. Install with: pip install datasets")


@dataclass
class TurnSample:
    """One LLM turn sample for PPO."""
    prompt_text: str
    response_text: str
    reward: float
    query_tensor: Optional[torch.Tensor] = None
    response_tensor: Optional[torch.Tensor] = None


@dataclass
class TraceEpisodeData:
    """Complete episode data for logging"""
    task_id: int
    seed: int
    prompt: str
    response: str
    ground_truth: str
    score: float
    transformed_code: str
    inputs: str
    dataset_index: int


class PrintInjector(ast.NodeTransformer):
    """Inject debug print statements into Python code"""
    
    def __init__(self, seed: int, max_injections: int = 6):
        self.rng = random.Random(seed)
        self.max_injections = max_injections
        self.injections_done = 0
        self.scopes: List[List[str]] = [[]]

    @property
    def live_vars(self) -> List[str]:
        all_vars = []
        for scope in self.scopes:
            for v in scope:
                if v not in all_vars:
                    all_vars.append(v)
        return all_vars

    def _add_live_var(self, name: str):
        if name not in self.scopes[-1]:
            self.scopes[-1].append(name)

    def _make_print(self) -> Optional[ast.Expr]:
        vars_available = self.live_vars
        if not vars_available:
            return None

        k = self.rng.randint(1, min(3, len(vars_available)))
        vars_to_print = self.rng.sample(vars_available, k)

        tag = f"__DBG_{self.injections_done}__"
        print_args = [ast.Constant(tag)]

        for var_name in vars_to_print:
            var_node = ast.Name(id=var_name, ctx=ast.Load())
            isinstance_check = ast.Call(
                func=ast.Name(id="isinstance", ctx=ast.Load()),
                args=[
                    var_node,
                    ast.Tuple(elts=[
                        ast.Name(id="int", ctx=ast.Load()),
                        ast.Name(id="float", ctx=ast.Load()),
                        ast.Name(id="str", ctx=ast.Load()),
                        ast.Name(id="bool", ctx=ast.Load()),
                        ast.Call(
                            func=ast.Name(id="type", ctx=ast.Load()),
                            args=[ast.Constant(None)],
                            keywords=[]
                        )
                    ], ctx=ast.Load())
                ],
                keywords=[]
            )
            repr_call = ast.Call(
                func=ast.Name(id="repr", ctx=ast.Load()),
                args=[ast.Name(id=var_name, ctx=ast.Load())],
                keywords=[]
            )
            type_name = ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id="type", ctx=ast.Load()),
                    args=[ast.Name(id=var_name, ctx=ast.Load())],
                    keywords=[]
                ),
                attr="__name__",
                ctx=ast.Load()
            )
            ternary = ast.IfExp(test=isinstance_check, body=repr_call, orelse=type_name)
            print_args.append(ternary)

        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=print_args,
                keywords=[]
            )
        )

    def _extract_names(self, target: ast.AST) -> Generator[str, None, None]:
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                yield from self._extract_names(elt)

    def _update_live_vars(self, stmt: ast.stmt):
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                for name in self._extract_names(t):
                    self._add_live_var(name)
        elif isinstance(stmt, ast.AugAssign):
            for name in self._extract_names(stmt.target):
                self._add_live_var(name)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            for name in self._extract_names(stmt.target):
                self._add_live_var(name)

    def _maybe_inject(self, body: List[ast.stmt]) -> List[ast.stmt]:
        new_body = []
        for stmt in body:
            self._update_live_vars(stmt)
            new_body.append(self.visit(stmt))

            if self.injections_done >= self.max_injections:
                continue

            if isinstance(stmt, (ast.Assign, ast.AugAssign)):
                if self.rng.random() < 0.6:
                    p = self._make_print()
                    if p:
                        new_body.append(p)
                        self.injections_done += 1
            elif isinstance(stmt, ast.If):
                if self.rng.random() < 0.4:
                    p = self._make_print()
                    if p:
                        new_body.append(p)
                        self.injections_done += 1

        return new_body

    def visit_Module(self, node: ast.Module):
        node.body = self._maybe_inject(node.body)
        return node

    def visit_For(self, node: ast.For):
        node.body = self._maybe_inject(node.body)
        node.orelse = self._maybe_inject(node.orelse)
        return node

    def visit_If(self, node: ast.If):
        node.body = self._maybe_inject(node.body)
        node.orelse = self._maybe_inject(node.orelse)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.scopes.append([])
        for arg in node.args.args:
            self._add_live_var(arg.arg)
        node.body = self._maybe_inject(node.body)
        self.scopes.pop()
        return node

    def visit_Assign(self, node: ast.Assign):
        return node

    def visit_AugAssign(self, node: ast.AugAssign):
        return node


def clean_source(source: str) -> str:
    """Strip markdown code blocks if present"""
    source = source.strip()
    if source.startswith("```"):
        first_newline = source.find("\n")
        if first_newline != -1:
            source = source[first_newline:].strip()
        if source.endswith("```"):
            source = source[:-3].strip()
    return source


def inject_prints(source: str, seed: int, max_injections: int = 6) -> str:
    """Inject debug print statements into Python code"""
    source = clean_source(source)
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"# Syntax Error in source: {e}\n{source}"

    injector = PrintInjector(seed=seed, max_injections=max_injections)
    tree = injector.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def run_code(code: str, input_data: str = "", timeout: int = 10) -> Tuple[str, Optional[str]]:
    """Execute code and capture stdout"""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr if result.returncode != 0 else None
    except subprocess.TimeoutExpired:
        return "", "Timeout"
    except Exception as e:
        return "", str(e)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def clean_prediction(prediction: str) -> str:
    """Remove reasoning tags and markdown from LLM response"""
    prediction = re.sub(r"<think>.*?</think>", "", prediction, flags=re.DOTALL)
    prediction = re.sub(r"<thinking>.*?</thinking>", "", prediction, flags=re.DOTALL)
    
    code_block_match = re.search(r"```(?:[a-zA-Z]*)\n?(.*?)\n?```", prediction, flags=re.DOTALL)
    if code_block_match:
        prediction = code_block_match.group(1)
    
    return prediction.strip()


def compare_outputs(expected: str, actual: str) -> bool:
    """Compare outputs with whitespace normalization"""
    def normalize(s):
        return "".join(s.split()).lower()
    return normalize(expected) == normalize(actual)


class LocalTraceTask:
    """Local Trace task generator and evaluator
    
    Supports two modes:
    1. Raw Python programs (satpalsr/rl-python) - generates challenges dynamically
    2. Pre-built SFT dataset (weirek/sft-warmup-trace) - uses pre-generated challenges
    """
    
    def __init__(
        self,
        dataset_name: str = "satpalsr/rl-python",
        dataset_config: str = None,
        dataset_split: str = "train",
        hf_token: str = None,
        use_sft_dataset: bool = False,
    ):
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets library not available")
        
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.use_sft_dataset = use_sft_dataset
        
        # Use SFT warmup dataset for consistency with SFT training
        if use_sft_dataset:
            dataset_name = "weirek/sft-warmup-trace"
            dataset_config = dataset_config or "chat_original"
            print(f"Loading SFT dataset: {dataset_name} config={dataset_config} split={dataset_split}")
            self.dataset = load_dataset(
                dataset_name, 
                dataset_config,
                split=dataset_split, 
                token=self.hf_token
            )
        else:
            print(f"Loading dataset: {dataset_name} split={dataset_split}")
            self.dataset = load_dataset(dataset_name, split=dataset_split, token=self.hf_token)
        
        print(f"Dataset loaded: {len(self.dataset)} examples")
    
    def generate_challenge(self, task_id: int = None, seed: int = None) -> Dict[str, Any]:
        """Generate a trace challenge
        
        If using SFT dataset, returns pre-built challenges.
        Otherwise, generates challenges dynamically with print injection.
        """
        if task_id is not None:
            idx = task_id % len(self.dataset)
            sample = self.dataset[idx]
        else:
            idx = random.randint(0, len(self.dataset) - 1)
            sample = self.dataset[idx]
        
        # Handle SFT dataset format (pre-built challenges)
        if self.use_sft_dataset:
            messages = sample.get("messages", [])
            if len(messages) >= 2:
                prompt = messages[0]["content"]  # User message
                ground_truth = messages[1]["content"]  # Assistant response
            else:
                prompt = ""
                ground_truth = ""
            
            return {
                "prompt": prompt,
                "ground_truth": ground_truth,
                "transformed_code": "",  # Not available in pre-built format
                "inputs": "",
                "seed": seed or 0,
                "dataset_index": idx,
                "task_id": task_id,
            }
        
        # Handle raw Python dataset (dynamic generation)
        source = sample.get("program", "")
        inputs = sample.get("inputs", "")
        
        if seed is None:
            seed = random.randint(0, 1000000)
        
        transformed = inject_prints(source, seed, max_injections=6)
        
        # Get ground truth
        stdout, stderr = run_code(transformed, input_data=inputs)
        ground_truth = stdout.strip()
        
        prompt = f"""Predict the exact and complete standard output (stdout) of the following Python program, including every single print statement.

The program contains several injected debug print statements starting with '__DBG_'. You must include these in your prediction exactly as they would appear in the output, along with any other output the program produces.

Program:
```python
{transformed}
```

Input (stdin):
```
{inputs}
```

Provide the full stdout content. Do not provide any explanations or commentary outside of the predicted output."""

        return {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "transformed_code": transformed,
            "inputs": inputs,
            "seed": seed,
            "dataset_index": idx,
            "task_id": task_id,
        }
    
    def evaluate_response(self, response: str, challenge: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate a response against the challenge"""
        ground_truth = challenge.get("ground_truth", "")
        cleaned = clean_prediction(response)
        
        score = 1.0 if compare_outputs(ground_truth, cleaned) else 0.0
        return score, "1/1" if score > 0 else "0/1"


def run_local_trace_episode(
    *,
    model,
    tokenizer,
    task_id: int,
    seed: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    max_seq_length: int = 2048,
    device: str = "cuda",
    trace_task: LocalTraceTask = None,
) -> Tuple[List[TurnSample], Dict[str, Any], TraceEpisodeData]:
    """
    Run a single Trace episode with local model inference.
    
    Args:
        model: The model to use for inference
        tokenizer: Tokenizer for the model
        task_id: Task ID for deterministic task selection
        seed: Random seed for LLM generation
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_new_tokens: Maximum tokens to generate
        max_seq_length: Maximum sequence length
        device: Device to run on
        trace_task: Pre-initialized LocalTraceTask (optional)
    
    Returns:
        turn_samples: List of TurnSample for PPO training
        episode_info: Dict with summary info
        episode_data: TraceEpisodeData with complete episode details
    """
    # Initialize trace task if not provided
    if trace_task is None:
        trace_task = LocalTraceTask()
    
    # Generate challenge
    challenge = trace_task.generate_challenge(task_id=task_id, seed=seed)
    prompt = challenge["prompt"]
    
    # Format as chat message
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_prompt = prompt
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    ).to(device)
    
    # Generate response
    with torch.no_grad():
        # Handle different model types
        if hasattr(model, 'pretrained_model'):
            gen_model = model.pretrained_model
        else:
            gen_model = model
        
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    
    # Evaluate
    score, test_result = trace_task.evaluate_response(response, challenge)
    
    # Create turn sample
    turn_samples = [TurnSample(
        prompt_text=prompt,
        response_text=response,
        reward=score,
    )]
    
    # Episode info
    episode_info = {
        "score": score,
        "success": score > 0,
        "task_id": task_id,
        "seed": seed,
        "test_result": test_result,
        "dataset_index": challenge["dataset_index"],
    }
    
    # Episode data for logging
    episode_data = TraceEpisodeData(
        task_id=task_id,
        seed=seed,
        prompt=prompt,
        response=response,
        ground_truth=challenge["ground_truth"],
        score=score,
        transformed_code=challenge["transformed_code"],
        inputs=challenge["inputs"],
        dataset_index=challenge["dataset_index"],
    )
    
    return turn_samples, episode_info, episode_data


# For testing
if __name__ == "__main__":
    # Test the trace task
    task = LocalTraceTask()
    
    # Generate a challenge
    challenge = task.generate_challenge(task_id=42)
    print("=" * 60)
    print("PROMPT:")
    print(challenge["prompt"][:500] + "...")
    print("=" * 60)
    print("GROUND TRUTH:")
    print(challenge["ground_truth"])
    print("=" * 60)
    
    # Test evaluation
    score, result = task.evaluate_response(challenge["ground_truth"], challenge)
    print(f"Self-evaluation score: {score} ({result})")
