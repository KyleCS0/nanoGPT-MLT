import os
import sys
import torch
import time
import json
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. 修正路徑問題 (確保能 import model)
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# 2. 配置 (Configuration)
# -----------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 確保這裡的模型參數與你的 GPT-2 設定一致
config = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768
)

# KV-cache toggle: set via command line or here
# v0 = False (no cache), v1 = True (with cache)
use_cache = True


# -----------------------------------------------------------------------------
# 3. 輔助函數 (Helper Functions)
# -----------------------------------------------------------------------------
def try_batch_size(model, batch_size, prompt_len, max_new_tokens, use_cache):
    """
    嘗試執行指定 batch size 的推理。
    Returns (success, peak_memory_bytes) 或 OOM 時返回 (False, 0)。
    """
    # CRITICAL: 每次測試前必須清空 Cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        # 建立 Dummy Data: shape (batch_size, prompt_len)
        idx = torch.randint(0, config.vocab_size, (batch_size, prompt_len), device=device)
        # 使用 torch.no_grad() 確保不計算梯度
        with torch.no_grad():
            model.generate(idx, max_new_tokens=max_new_tokens, use_cache=use_cache)
        # 取得峰值記憶體 (不是當前記憶體，而是過程中最大值)
        peak_mem = torch.cuda.max_memory_allocated()
        return True, peak_mem
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # 清理 OOM 狀態
            torch.cuda.empty_cache()
            return False, 0
        raise


def find_max_batch_binary(model, prompt_len, max_new_tokens, use_cache,
                          initial_high=512, verbose=True):
    """
    使用二分搜尋找到精確的最大 batch size。

    策略：
    1. 先用指數搜尋找到上界 (OOM 發生的位置)
    2. 再用二分搜尋在最後成功和首次失敗之間精確定位

    Returns: (max_batch_size, peak_memory_at_max)
    """
    # Phase 1: 指數搜尋找到粗略的上界
    if verbose:
        print("Phase 1: 尋找上界 (指數搜尋)...")

    low = 1
    high = 1
    last_success = 1
    last_success_mem = 0

    # 指數成長直到 OOM
    while high <= initial_high:
        if verbose:
            print(f"  測試 batch={high}...", end="", flush=True)

        success, peak_mem = try_batch_size(model, high, prompt_len, max_new_tokens, use_cache)

        if success:
            if verbose:
                print(f" [OK] Peak: {peak_mem/1024**3:.2f} GB")
            last_success = high
            last_success_mem = peak_mem
            low = high
            high *= 2
        else:
            if verbose:
                print(f" [OOM]")
            break

    # 如果從未 OOM，返回最高測試值
    if high > initial_high:
        if verbose:
            print(f"  已達搜尋上限 ({initial_high})")
        return last_success, last_success_mem

    # Phase 2: 二分搜尋找到精確值
    if verbose:
        print(f"\nPhase 2: 二分搜尋 範圍 [{low}, {high}]...")

    best = last_success
    best_mem = last_success_mem

    while low < high - 1:
        mid = (low + high) // 2

        if verbose:
            print(f"  測試 batch={mid}...", end="", flush=True)

        success, peak_mem = try_batch_size(model, mid, prompt_len, max_new_tokens, use_cache)

        if success:
            if verbose:
                print(f" [OK] Peak: {peak_mem/1024**3:.2f} GB")
            best = mid
            best_mem = peak_mem
            low = mid
        else:
            if verbose:
                print(f" [OOM]")
            high = mid

    return best, best_mem


# -----------------------------------------------------------------------------
# 4. 主要測試函數
# -----------------------------------------------------------------------------
def run_inference_stress_test(use_cache=True, use_binary_search=True):
    """
    執行推理容量壓力測試。

    Args:
        use_cache: 是否使用 KV-cache
        use_binary_search: 是否使用二分搜尋找到精確的最大值 (預設 True)
    """
    version = "v1 (KV-cache)" if use_cache else "v0 (no cache)"
    print(f"Initializing Model on {device}...")
    model = GPT(config)
    model.to(device)
    model.eval() # 設定為評估模式 (Inference Mode)

    print(f"Starting Inference Capacity Stress Test - {version}")
    print(f"use_cache={use_cache}")
    print("-" * 50)

    # 建立一個假的輸入 Prompt (隨機整數)
    # 假設我們給定一半的 block_size 作為 prompt，讓它生成剩下的
    prompt_len = 512
    max_new_tokens = 50

    if use_binary_search:
        # 使用二分搜尋找到精確的最大 batch size
        max_capacity, peak_mem = find_max_batch_binary(
            model, prompt_len, max_new_tokens, use_cache,
            initial_high=512, verbose=True
        )
        print("-" * 50)
        print(f"Final Max Capacity (Inference): Batch Size {max_capacity}")
        print(f"Peak Memory at Max Batch: {peak_mem/1024**3:.2f} GB")
        return max_capacity, peak_mem
    else:
        # 原始方法：依照描述，測試 2 的次方
        batch_sizes = 1
        max_capacity = 0

        while True:
            try:
                # CRITICAL: 依照描述，每次循環開頭必須清空 Cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                print(f"Testing Batch Size {batch_sizes}...", end="", flush=True)

                # 建立 Dummy Data: shape (batch_size, prompt_len)
                # 使用 randint 模擬輸入 token
                idx = torch.randint(0, config.vocab_size, (batch_sizes, prompt_len), device=device)

                # 依照描述：生成 50 個 token
                # 這裡使用 torch.no_grad() 確保不計算梯度 (雖然 model.generate 內部通常會處理，但顯式寫出更好)
                with torch.no_grad():
                    model.generate(idx, max_new_tokens=max_new_tokens, use_cache=use_cache)

                # 如果執行到這裡沒報錯，代表成功
                # 使用峰值記憶體而非當前記憶體
                peak_mem = torch.cuda.max_memory_allocated()
                print(f" [OK] | Peak Memory: {peak_mem/1024**3:.2f} GB")
                max_capacity = batch_sizes
                batch_sizes *= 2  # 測試下一個 2 的次方

            except RuntimeError as e:
                # 捕捉 OOM 錯誤
                if "out of memory" in str(e):
                    print(f"\n[CRASH] OOM reached at Batch Size: {batch_sizes}!")
                    print("-" * 50)
                    print(f"Final Max Capacity (Inference): Batch Size {max_capacity}")
                    break
                else:
                    # 如果是其他錯誤，印出來
                    print(f"\n[ERROR] {e}")
                    raise e
        return max_capacity, 0


def run_capacity_comparison(versions_to_test=['v0', 'v1']):
    """
    執行多版本容量測試，並報告相對改善比例。

    這個函數的學術意義：
    - 雖然峰值記憶體已經顯示理論上的增益
    - 此測試驗證這些增益在實際 CUDA allocator 行為下轉化為真實的服務容量
    - 報告相對比例 (V1/V0)，使結果與硬體無關
    """
    print(f"Initializing Model on {device}...")
    print(f"Model: {config.n_layer}L, {config.n_head}H, {config.n_embd}D")
    print(f"Test config: prompt_len=512, max_new_tokens=50")
    print("=" * 60)

    results = {}
    prompt_len = 512
    max_new_tokens = 50

    for version in versions_to_test:
        use_cache = version != 'v0'
        version_name = f"{version} ({'KV-cache' if use_cache else 'no cache'})"

        print(f"\n{'='*60}")
        print(f"Testing {version_name}")
        print("=" * 60)

        # 每個版本重新初始化模型，確保乾淨狀態
        model = GPT(config)
        model.to(device)
        model.eval()

        max_batch, peak_mem = find_max_batch_binary(
            model, prompt_len, max_new_tokens, use_cache,
            initial_high=512, verbose=True
        )

        results[version] = {
            'max_batch': max_batch,
            'peak_memory_gb': peak_mem / 1024**3,
            'use_cache': use_cache
        }

        # 清理
        del model
        torch.cuda.empty_cache()

    # 印出比較表格
    print("\n" + "=" * 60)
    print("RESULTS: Inference Batch Capacity")
    print("(prompt_len=512, max_new_tokens=50)")
    print("=" * 60)

    # 決定基準線 (v0 如果存在，否則第一個版本)
    baseline_version = 'v0' if 'v0' in results else list(results.keys())[0]
    baseline_batch = results[baseline_version]['max_batch']

    print(f"\n{'Version':<12} {'Max Batch':>10} {'Peak Mem (GB)':>14} {'vs Baseline':>12}")
    print("-" * 50)

    for version, data in results.items():
        ratio = data['max_batch'] / baseline_batch
        marker = " (baseline)" if version == baseline_version else ""
        print(f"{version:<12} {data['max_batch']:>10} {data['peak_memory_gb']:>14.2f} {ratio:>11.2f}×{marker}")

    # 儲存結果到 JSON
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'block_size': config.block_size,
            'prompt_len': prompt_len,
            'max_new_tokens': max_new_tokens
        },
        'device': torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
        'results': results,
        'baseline_version': baseline_version
    }

    output_path = os.path.join(current_dir, 'capacity_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference capacity stress test (支援二分搜尋)")
    parser.add_argument('--version', type=str, default='v1', choices=['v0', 'v1', 'all'],
                        help='v0 = no cache, v1 = KV-cache, all = 比較兩者 (default: v1)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable KV-cache (same as --version v0)')
    parser.add_argument('--compare', action='store_true',
                        help='執行 v0 和 v1 的比較測試 (same as --version all)')
    parser.add_argument('--legacy', action='store_true',
                        help='使用原始的 2 的次方搜尋方法 (不使用二分搜尋)')
    args = parser.parse_args()

    if device == 'cpu':
        print("Warning: CUDA not found. Testing on CPU will be slow and won't measure VRAM.")
        sys.exit(1)

    # 決定執行模式
    if args.compare or args.version == 'all':
        # 比較模式：測試 v0 和 v1 並報告相對改善
        run_capacity_comparison(['v0', 'v1'])
    else:
        # 單版本模式
        use_cache = not args.no_cache and args.version != 'v0'
        use_binary_search = not args.legacy
        run_inference_stress_test(use_cache=use_cache, use_binary_search=use_binary_search)