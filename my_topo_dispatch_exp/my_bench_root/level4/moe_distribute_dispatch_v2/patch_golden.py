with open("golden.py", "r") as f:
    code = f.read()

old_func = '''def _make_tensor(shape, dtype_name: str, value_range, seed: int, device):
    torch.manual_seed(int(seed))
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    dtype = dtype_map.get(str(dtype_name).lower(), torch.float32)
    lo, hi = value_range if value_range is not None else [0, 1]
    if dtype.is_floating_point:
        tensor = torch.empty(shape, dtype=torch.float32).uniform_(float(lo), float(hi)).to(dtype)
    else:
        tensor = torch.randint(int(lo), int(hi) + 1, shape, dtype=dtype)
    return tensor.to(device)'''

new_func = '''def _make_tensor(shape, dtype_name: str, value_range, seed: int, device):
    import os
    torch.manual_seed(int(seed))
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    dtype = dtype_map.get(str(dtype_name).lower(), torch.float32)
    lo, hi = value_range if value_range is not None else [0, 1]
    if dtype.is_floating_point:
        tensor = torch.empty(shape, dtype=torch.float32).uniform_(float(lo), float(hi)).to(dtype)
    else:
        alpha = float(os.environ.get("GRAPH_SKEW_ALPHA", "0.0"))
        if alpha > 0.0 and len(shape) == 2:
            # 模拟图计算/科学计算中的幂律分布 (Power-Law / Zipf Skew)
            num_classes = int(hi) - int(lo) + 1
            ranks = torch.arange(1, num_classes + 1, dtype=torch.float32)
            probs = 1.0 / (ranks ** alpha)
            probs = probs / probs.sum()
            # 无放回按权重采样，保证同一 Token 的 Top-K 专家互不重复
            bs, k = shape[0], shape[1]
            sampled = torch.multinomial(probs.expand(bs, -1), num_samples=k, replacement=False)
            tensor = (sampled + int(lo)).to(dtype)
        else:
            tensor = torch.randint(int(lo), int(hi) + 1, shape, dtype=dtype)
    return tensor.to(device)'''

assert old_func in code, "Target function not found!"
with open("golden.py", "w") as f:
    f.write(code.replace(old_func, new_func))
print("Successfully patched golden.py with GRAPH_SKEW_ALPHA support!")
