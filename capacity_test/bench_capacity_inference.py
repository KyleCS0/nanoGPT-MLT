import os
import sys
import torch
import time

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

def run_inference_stress_test():
    print(f"Initializing Model on {device}...")
    model = GPT(config)
    model.to(device)
    model.eval() # 設定為評估模式 (Inference Mode)
    
    print("Starting Inference Capacity Stress Test (Powers of 2)...")
    print("-" * 50)
    
    # 依照描述：測試 2 的次方
    batch_sizes = 1
    max_capacity = 0
    
    # 建立一個假的輸入 Prompt (隨機整數)
    # 假設我們給定一半的 block_size 作為 prompt，讓它生成剩下的
    prompt_len = 512 
    
    while True:
        try:
            # CRITICAL: 依照描述，每次循環開頭必須清空 Cache
            torch.cuda.empty_cache()
            
            print(f"Testing Batch Size {batch_sizes}...", end="", flush=True)
            
            # 建立 Dummy Data: shape (batch_size, prompt_len)
            # 使用 randint 模擬輸入 token
            idx = torch.randint(0, config.vocab_size, (batch_sizes, prompt_len), device=device)
            
            # 依照描述：生成 50 個 token
            # 這裡使用 torch.no_grad() 確保不計算梯度 (雖然 model.generate 內部通常會處理，但顯式寫出更好)
            with torch.no_grad():
                model.generate(idx, max_new_tokens=50)
            
            # 如果執行到這裡沒報錯，代表成功
            print(f" [OK] | Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
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

if __name__ == "__main__":
    if device == 'cpu':
        print("Warning: CUDA not found. Testing on CPU will be slow and won't measure VRAM.")
    
    run_inference_stress_test()