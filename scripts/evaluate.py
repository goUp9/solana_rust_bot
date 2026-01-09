#!/usr/bin/env python3
"""
Quick evaluation script for trained models
"""

import asyncio
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from affine.core.environments import create_environment


async def evaluate_model(
    checkpoint_path: str,
    num_episodes: int = 50,
    env_mode: str = "basilica",
    opponent: str = "random",
):
    """
    Evaluate a trained model
    
    Args:
        checkpoint_path: Path to checkpoint directory
        num_episodes: Number of episodes to evaluate
        env_mode: Environment mode (docker or basilica)
        opponent: Opponent type (random or mcts)
    """
    checkpoint_path = Path(checkpoint_path)
    
    print("Loading model...")
    
    # Load metadata
    with open(checkpoint_path / "metadata.json") as f:
        metadata = json.load(f)
    
    base_model_name = metadata.get("base_model", "Qwen/Qwen2.5-3B-Instruct")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path / "tokenizer",
        trust_remote_code=True
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path / "model"
    )
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Base model: {base_model_name}")
    
    # Initialize environment
    print(f"Initializing environment (mode={env_mode})...")
    env = create_environment("game", mode=env_mode)
    
    # Run evaluation
    print(f"\nRunning {num_episodes} evaluation episodes...")
    results = []
    
    for i in range(num_episodes):
        # Sample random task
        task_id = torch.randint(0, 1000000000, (1,)).item()
        seed = torch.randint(0, 2**31, (1,)).item()
        
        try:
            # This is a simplified version - in practice you'd need to set up
            # a local inference server or integrate the model directly
            result = await env.evaluate(
                miner=None,  # Would need to mock this
                task_id=task_id,
                seed=seed,
                opponent=opponent,
            )
            
            results.append({
                "task_id": task_id,
                "score": result.score,
                "success": result.success,
            })
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{num_episodes}")
                
        except Exception as e:
            print(f"Error on episode {i + 1}: {e}")
            continue
    
    # Compute statistics
    if results:
        mean_score = sum(r["score"] for r in results) / len(results)
        success_rate = sum(r["success"] for r in results) / len(results)
        
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        print(f"Episodes: {len(results)}")
        print(f"Mean Score: {mean_score:.4f}")
        print(f"Success Rate: {success_rate:.2%}")
        print("="*60)
        
        # Save results
        results_path = checkpoint_path / "eval_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "num_episodes": len(results),
                "mean_score": mean_score,
                "success_rate": success_rate,
                "results": results,
            }, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
    else:
        print("\nNo successful episodes!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--env_mode",
        type=str,
        default="basilica",
        choices=["docker", "basilica"],
        help="Environment mode"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "mcts"],
        help="Opponent type"
    )
    
    args = parser.parse_args()
    
    asyncio.run(evaluate_model(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        env_mode=args.env_mode,
        opponent=args.opponent,
    ))


if __name__ == "__main__":
    main()
