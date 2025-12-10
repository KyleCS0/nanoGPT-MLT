"""
bench_capacity_training.py
用於測試 nanoGPT 在當前 GPU 硬體下的最大 Batch Size 容量。
"""
import sys
import os
import math
import time
import torch

# --- 新增這段程式碼來解決路徑問題 ---
# 取得目前檔案所在的資料夾路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
# 取得上一層資料夾路徑 (即 nanoGPT-MLT 根目錄)
parent_dir = os.path.dirname(current_dir)
# 將上一層加入系統搜尋路徑
sys.path.append(parent_dir)

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# 配置 (Configuration)
# 你可以根據想要測試的模型大小修改這裡
# 預設使用 GPT-2 (124M parameters) 的配置
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

# 系統設定
device = 'cuda' # 通常我們只在 cuda 上測試
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16'
compile = False # 是否使用 torch.compile (會增加編譯時間，但可能改變記憶體使用)
# -----------------------------------------------------------------------------

# 設定 PyTorch
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def run_stress_test():
    # 1. 初始化模型
    print(f"Initializing GPT model (n_layer={n_layer}, n_embd={n_embd})...")
    config = GPTConfig(
        block_size=block_size,
        vocab_size=50304, # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )
    model = GPT(config)
    model.to(device)
    
    if compile:
        print("Compiling model... (this might take a minute)")
        model = torch.compile(model)

    print("Model initialized. Starting capacity stress test...")
    print(f"{'-'*40}")
    print(f"{'Batch Size':<15} | {'Status':<10} | {'Memory (Allocated)'}")
    print(f"{'-'*40}")

    batch_size = 1
    max_successful_batch = 0
    
    # 用於 Backward pass 的 Scaler (如果使用 float16)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    while True:
        try:
            # 強制清理快取，確保測試獨立性
            torch.cuda.empty_cache()
            
            # 2. 建立 Dummy Data (隨機整數)
            # 形狀: (batch_size, block_size)
            x = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)
            y = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)

            # 3. 執行 Forward + Backward Pass
            # 我們必須跑 Backward，因為梯度 (Gradients) 會佔用大量記憶體
            with ctx:
                _, loss = model(x, y)
            
            # 模擬 Backward pass
            scaler.scale(loss).backward()
            
            # 如果程式跑到這裡沒有報錯，代表成功
            mem_usage = torch.cuda.memory_allocated() / 1024**3 # in GB
            print(f"{batch_size:<15} | {'OK':<10} | {mem_usage:.2f} GB")
            
            max_successful_batch = batch_size
            
            # 清空梯度，準備下一次迴圈
            model.zero_grad(set_to_none=True)
            
            # 增加 Batch Size
            # 你可以改為 batch_size += 1 (線性) 或 batch_size *= 2 (指數)
            # 這裡使用線性增加以找出精確極限
            batch_size += 1

        except torch.cuda.OutOfMemoryError:
            print(f"{'-'*40}")
            print(f"!!! OOM (Out Of Memory) reached at Batch Size: {batch_size} !!!")
            print(f"Maximum stable Batch Size: {max_successful_batch}")
            break
        except RuntimeError as e:
            # 有些舊版 PyTorch OOM 會在 RuntimeError 裡
            if "out of memory" in str(e):
                print(f"{'-'*40}")
                print(f"!!! OOM reached at Batch Size: {batch_size} !!!")
                print(f"Maximum stable Batch Size: {max_successful_batch}")
                break
            else:
                raise e

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires a GPU.")
    else:
        run_stress_test()