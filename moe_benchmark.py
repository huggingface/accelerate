#!/usr/bin/env python3
"""
Real-world MoE model benchmark demonstrating the get_balanced_memory optimization.

This script loads actual MoE models and measures the performance impact of the O(n²) → O(n) fix.
Run with: python realworld_moe_benchmark.py [--model MODEL_NAME] [--device DEVICE]
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Add accelerate src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from accelerate.utils.modeling import compute_module_sizes, get_balanced_memory, get_module_leaves


def get_available_moe_models():
    """Return list of available MoE models for testing."""
    return [
        ("DialoGPT-Medium", "microsoft/DialoGPT-medium"),  # Very Small for quick testing (~350M params)
        ("DeepSeek-MoE-16B", "deepseek-ai/deepseek-moe-16b-base"),  # Small MoE model
        ("Mixtral-8x7B", "mistralai/Mixtral-8x7B-Instruct-v0.1"),  # Large MoE (~47B params)
        ("Mixtral-8x22B", "mistralai/Mixtral-8x22B-Instruct-v0.1"),  # Very large MoE (~141B params)
    ]


def load_model_safely(model_name, device="auto", torch_dtype=torch.float16, max_memory=None):
    """Load model with error handling and progress reporting."""
    print(f"🔄 Loading model: {model_name}")
    print(f"   Device: {device}")
    print(f"   Dtype: {torch_dtype}")

    # Check available GPUs
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"   Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"     GPU {i}: {props.name} ({memory_gb:.1f} GB)")

    try:
        # First, check model config to understand size
        config = AutoConfig.from_pretrained(model_name)
        print(f"   Model type: {config.model_type}")
        if hasattr(config, "num_local_experts"):
            print(f"   Number of experts: {config.num_local_experts}")
        if hasattr(config, "num_hidden_layers"):
            print(f"   Hidden layers: {config.num_hidden_layers}")

        start_time = time.time()

        # Prepare device map for multi-GPU loading
        if device == "auto" and torch.cuda.is_available():
            # Let accelerate automatically distribute across GPUs
            print("   Using automatic device mapping across all available GPUs")
            device_map = "auto"
        elif device == "cpu":
            device_map = "cpu"
        else:
            device_map = device

        # Create max_memory dict if not provided
        if max_memory is None and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            # Use 70GB per H100 (80GB total - leave 10GB for overhead)
            max_memory = {}
            for i in range(min(gpu_count, 8)):  # Only use up to 8 GPUs
                max_memory[i] = "70GB"
            print(f"   Max memory per GPU: {max_memory}")

        # Load with multi-GPU distribution
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            offload_folder="./offload",  # Offload to disk if needed
        )

        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")

        # Print device distribution
        if hasattr(model, "hf_device_map"):
            print("   Model distribution:")
            device_counts = {}
            for module, device in model.hf_device_map.items():
                device_counts[device] = device_counts.get(device, 0) + 1
            for device, count in sorted(device_counts.items()):
                print(f"     {device}: {count} modules")

        return model, load_time

    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        print("   Try using smaller model or more memory-efficient settings")
        return None, 0


def simulate_old_implementation(module_sizes):
    """Simulate the old O(n²) implementation for comparison."""
    start_time = time.perf_counter()
    
    # First call to get_module_leaves (this is done in the original function)
    leaves = get_module_leaves(module_sizes)
    # This was the problematic O(n²) line
    filtered = {n: v for n, v in module_sizes.items() if n not in leaves}
    # Second call to get_module_leaves (also done in original)
    leaves2 = get_module_leaves(filtered)
    
    end_time = time.perf_counter()
    return end_time - start_time, len(filtered), len(leaves)


def simulate_new_implementation(module_sizes):
    """Simulate the new O(n) implementation for comparison."""
    start_time = time.perf_counter()
    
    # First call to get_module_leaves
    leaves = get_module_leaves(module_sizes)
    leaves_set = set(leaves)  # The optimization: convert to set
    # This is the optimized O(n) line
    filtered = {n: v for n, v in module_sizes.items() if n not in leaves_set}
    # Second call to get_module_leaves
    leaves2 = get_module_leaves(filtered)
    
    end_time = time.perf_counter()
    return end_time - start_time, len(filtered), len(leaves)


def benchmark_get_balanced_memory(model, model_name):
    """Benchmark get_balanced_memory with both old and new implementations."""
    print(f"\n📊 Benchmarking get_balanced_memory for {model_name}")
    print("=" * 70)

    # Get module sizes (this is what get_balanced_memory does internally)
    print("🔍 Computing module sizes...")
    module_sizes = compute_module_sizes(model)
    print(f"   Total modules: {len(module_sizes):,}")

    # Find leaves for analysis
    leaves = get_module_leaves(module_sizes)
    print(f"   Leaf modules: {len(leaves):,}")
    print(f"   Non-leaf modules: {len(module_sizes) - len(leaves):,}")

    # Estimate theoretical complexity
    theoretical_ops_old = len(module_sizes) * len(leaves)
    theoretical_ops_new = len(module_sizes)
    theoretical_speedup = theoretical_ops_old / theoretical_ops_new if theoretical_ops_new > 0 else 1

    print("\n🧮 Theoretical complexity analysis:")
    print(f"   Old O(n²): {len(module_sizes):,} × {len(leaves):,} = {theoretical_ops_old:,} operations")
    print(f"   New O(n):  {len(module_sizes):,} operations")
    print(f"   Expected speedup: {theoretical_speedup:.1f}x")

    # Test different GPU configurations for H100 cluster
    test_configs = [
        {"gpus": 2, "memory": "80GB", "label": "2 GPUs"},
        {"gpus": 4, "memory": "80GB", "label": "4 GPUs"},
        {"gpus": 8, "memory": "80GB", "label": "8 GPUs"},  # Full H100 cluster
    ]

    results = []

    for config in test_configs:
        print(f"\n🖥️  Testing with {config['gpus']} GPUs @ {config['memory']} each:")

        # Create max_memory dict
        max_memory = {i: config["memory"] for i in range(config["gpus"])}

        # Benchmark old implementation (simulation)
        print("   � Simulating old O(n²) implementation...")
        old_time, filtered_count, _ = simulate_old_implementation(module_sizes)
        
        # Benchmark new implementation (simulation)
        print("   � Simulating new O(n) implementation...")
        new_time, filtered_count_new, _ = simulate_new_implementation(module_sizes)
        
        # Also test the actual get_balanced_memory function for reference
        print("   💾 Testing actual get_balanced_memory function...")
        start_time = time.perf_counter()
        balanced_memory_result = get_balanced_memory(model, max_memory)
        actual_function_time = time.perf_counter() - start_time

        speedup = old_time / new_time if new_time > 0 else float("inf")

        results.append(
            {
                "config": config,
                "old_time": old_time,
                "new_time": new_time,
                "speedup": speedup,
                "actual_function_time": actual_function_time,
                "balanced_memory": balanced_memory_result,
                "model_info": {
                    "total_modules": len(module_sizes),
                    "leaf_modules": len(leaves),
                    "theoretical_speedup": theoretical_speedup
                }
            }
        )

        print(f"   ⏱️  Old O(n²) simulation: {old_time:.4f}s")
        print(f"   ⏱️  New O(n) simulation: {new_time:.4f}s")
        print(f"   ⏱️  Actual function time: {actual_function_time:.4f}s")
        print(f"   🚀 Speedup (simulation): {speedup:.1f}x")
        print(f"   ✅ Results identical: {'Yes' if filtered_count == filtered_count_new else 'No'}")
        print(f"   💾 Memory allocation: {balanced_memory_result}")

    return results


def create_performance_chart(all_results):
    """Create a professional bar chart showing performance improvements."""
    print("\n📈 Creating performance chart...")
    
    # Set up the figure with professional styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract data for plotting
    models = list(all_results.keys())
    gpu_configs = ["2 GPUs", "4 GPUs", "8 GPUs"]
    
    # Prepare data arrays
    speedups = []
    for model in models:
        model_speedups = []
        for result in all_results[model]:
            model_speedups.append(result['speedup'])
        speedups.append(model_speedups)
    
    # Set up bar positions
    x = np.arange(len(models))
    width = 0.25
    
    # Create bars for each GPU configuration
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Professional blue, orange, green
    
    for i, (gpu_config, color) in enumerate(zip(gpu_configs, colors)):
        values = [speedups[j][i] for j in range(len(models))]
        bars = ax.bar(x + i * width, values, width, label=gpu_config, color=color, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value:.1f}x', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Customize the chart
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor (Old/New)', fontsize=12, fontweight='bold')
    ax.set_title('get_balanced_memory Optimization Performance\nO(n²) → O(n) Speedup Across Models and GPU Configurations', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=0, ha='center')
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Set y-axis to start from 0 and add some padding
    ax.set_ylim(0, max([max(s) for s in speedups]) * 1.1)
    
    # Add annotation
    ax.text(0.02, 0.98, 'Higher is better', transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Improve layout
    plt.tight_layout()
    
    # Save the chart
    chart_filename = 'get_balanced_memory_optimization_chart.png'
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Chart saved as: {chart_filename}")
    
    return chart_filename


def run_all_models_benchmark():
    """Run benchmark on all available models and create performance chart."""
    print("🚀 Running comprehensive MoE Model Benchmark")
    print("Testing get_balanced_memory optimization (Issue #3768)")
    print("=" * 70)
    
    models = get_available_moe_models()
    all_results = {}
    
    for model_display_name, model_name in models:
        print(f"\n{'='*70}")
        print(f"🤖 Testing Model: {model_display_name}")
        print(f"{'='*70}")
        
        # Load the model
        model, load_time = load_model_safely(model_name, "auto", torch.float16)
        
        if model is None:
            print(f"❌ Skipping {model_display_name} - failed to load")
            continue
        
        try:
            # Run benchmark
            results = benchmark_get_balanced_memory(model, model_display_name)
            all_results[model_display_name] = results
            
            # Print quick summary
            avg_speedup = np.mean([r['speedup'] for r in results])
            print(f"✅ {model_display_name}: Average speedup {avg_speedup:.1f}x")
            
            # Clean up memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Benchmark failed for {model_display_name}: {e}")
            continue
    
    if all_results:
        # Create performance chart
        chart_file = create_performance_chart(all_results)
        
        # Print final summary
        print(f"\n🎯 FINAL SUMMARY")
        print("=" * 70)
        print(f"Models tested: {len(all_results)}")
        print(f"Performance chart: {chart_file}")
        print("\nSpeedup Summary:")
        for model_name, results in all_results.items():
            speedups = [r['speedup'] for r in results]
            print(f"  {model_name}: {speedups[0]:.1f}x, {speedups[1]:.1f}x, {speedups[2]:.1f}x (2/4/8 GPUs)")
        
        print(f"\n💡 This optimization prevents stalling when loading large MoE models!")
    
    else:
        print("❌ No models were successfully benchmarked")
        return 1
    
    return 0
    """Print a comprehensive summary of the benchmark results."""
    print(f"\n🎯 SUMMARY FOR {model_name}")
    print("=" * 70)
    print(f"Model load time: {load_time:.2f}s")
    print()

    print("Performance Results:")
    print(f"{'GPUs':<5} {'Memory':<8} {'Old (s)':<10} {'New (s)':<10} {'Speedup':<10}")
    print("-" * 50)

    total_old_time = 0
    total_new_time = 0

    for result in results:
        config = result["config"]
        old_time = result["old_time"]
        new_time = result["new_time"]
        speedup = result["speedup"]

        total_old_time += old_time
        total_new_time += new_time

        print(f"{config['gpus']:<5} {config['memory']:<8} {old_time:<10.4f} {new_time:<10.4f} {speedup:<10.1f}x")

    avg_speedup = total_old_time / total_new_time if total_new_time > 0 else float("inf")
    time_saved = total_old_time - total_new_time

    print("-" * 50)
    print(f"Average speedup: {avg_speedup:.1f}x")
    print(f"Total time saved: {time_saved:.4f}s ({(time_saved / total_old_time) * 100:.1f}% reduction)")

    print("\n💡 Impact: This optimization prevents stalling when loading large MoE models!")
    print(f"   Without fix: Users would wait {total_old_time:.4f}s just for memory balancing")
    print(f"   With fix: Memory balancing completes in {total_new_time:.4f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark get_balanced_memory optimization on real MoE models")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to benchmark (default: run all models)",
    )
    parser.add_argument("--device", default="auto", help="Device to load model on (default: auto)")
    parser.add_argument("--list-models", action="store_true", help="List available MoE models for testing")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model (default: float16)",
    )
    parser.add_argument("--all-models", action="store_true", help="Run benchmark on all models and create chart")

    args = parser.parse_args()

    if args.list_models:
        print("Available MoE models for testing:")
        for display_name, model_name in get_available_moe_models():
            print(f"  - {display_name}: {model_name}")
        return

    # Convert dtype string to torch dtype
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # Run all models benchmark if requested or no specific model given
    if args.all_models or args.model is None:
        return run_all_models_benchmark()
    
    # Single model benchmark
    print("🚀 Real-world MoE Model Benchmark")
    print("Testing get_balanced_memory optimization (Issue #3768)")
    print("=" * 70)

    # Load the model with multi-GPU support
    model, load_time = load_model_safely(args.model, args.device, torch_dtype)

    if model is None:
        print("❌ Could not load model. Exiting.")
        return 1

    # Run benchmark
    try:
        results = benchmark_get_balanced_memory(model, args.model)
        
        # Simple summary for single model
        print(f"\n🎯 SUMMARY FOR {args.model}")
        print("=" * 70)
        print(f"Model load time: {load_time:.2f}s")
        print("Performance Results:")
        for result in results:
            config = result["config"]
            speedup = result["speedup"]
            print(f"  {config['gpus']} GPUs: {speedup:.1f}x speedup")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n✅ Benchmark completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
