import os, glob

print("=== 1. 目录文件结构 ===")
for f in sorted(glob.glob("**/*", recursive=True)):
    if os.path.isfile(f) and not f.endswith((".so", ".o", ".bin", ".pyc")):
        print(" ", f)

print("\n=== 2. run_2npu_bench.py 校验逻辑 ===")
if os.path.exists("run_2npu_bench.py"):
    with open("run_2npu_bench.py", "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        if any(k in line for k in ["MaxDiff", "hier", "Pass", "golden", "cand"]):
            start = max(0, i - 3)
            end = min(len(lines), i + 4)
            print(f"--- Line {i} ---")
            for j in range(start, end):
                print(f"{j+1:4d}: {lines[j]}", end="")
            print()
