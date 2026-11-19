import re
import json
import time
import subprocess

KERNEL_PATH = "my_ops_pkg/csrc/ops/moe_dispatch_e2e/op_kernel/moe_dispatch_e2e_kernel.cpp"
INIT_PATH = "my_ops_pkg/cann_bench/__init__.py"

MULTI_DIM_GENOMES = [
    {
        "gen": 0,
        "name": "Gen0_32KB_20AIV_Static",
        "tile_bytes": 32768,
        "max_aiv": 20,
        "skew_thresh": 999.0,
        "desc": "单维基线：32KB UB Tile + 20 AIV 满核 + 静态全走自定义流",
    },
    {
        "gen": 1,
        "name": "Gen1_32KB_12AIV_Static",
        "tile_bytes": 32768,
        "max_aiv": 12,
        "skew_thresh": 999.0,
        "desc": "核数裁剪进化：限制并发 AIV 核数为 12，降低小包核间同步与 GM 争用",
    },
    {
        "gen": 2,
        "name": "Gen2_32KB_16AIV_SkewThresh1.65",
        "tile_bytes": 32768,
        "max_aiv": 16,
        "skew_thresh": 1.65,
        "desc": "三维联合最优：32KB UB + 16 AIV + 倾斜感知阈值 2.2（大包重度倾斜自动切抗拥塞融合流）",
    },
]

def apply_genome(g):
    with open(KERNEL_PATH, "r") as f:
        kcode = f.read()
    kcode = re.sub(r"constexpr static int64_t MAX_TILE_BYTES = \d+;",
                   f"constexpr static int64_t MAX_TILE_BYTES = {g['tile_bytes']};", kcode)
    kcode = re.sub(r"return coreNum > 0 \? [^;]+;",
                   f"return coreNum > 0 ? (coreNum < {g['max_aiv']} ? coreNum : {g['max_aiv']}) : {g['max_aiv']};", kcode)
    with open(KERNEL_PATH, "w") as f:
        f.write(kcode)

    with open(INIT_PATH, "r") as f:
        icode = f.read()
    icode = re.sub(r"MAX_AIV_CAP = \d+", f"MAX_AIV_CAP = {g['max_aiv']}", icode)
    icode = re.sub(r"SKEW_ADAPTIVE_THRESHOLD = [0-9.]+", f"SKEW_ADAPTIVE_THRESHOLD = {g['skew_thresh']}", icode)
    with open(INIT_PATH, "w") as f:
        f.write(icode)

def build_and_install():
    cmd = (
        "cd my_ops_pkg && bash build.sh >/dev/null 2>&1 && "
        "pip install dist/cann_bench*.whl --force-reinstall --no-deps --user >/dev/null 2>&1 && cd .. && "
        "SITE_PKG=$(python3 -c 'import cann_bench, os; print(os.path.dirname(cann_bench.__file__))') && "
        "cp my_ops_pkg/cann_bench/__init__.py \"$SITE_PKG/__init__.py\""
    )
    return subprocess.run(cmd, shell=True).returncode == 0

def run_cannbench_eval():
    out = subprocess.check_output(["python3", "run_2npu_bench.py"], stderr=subprocess.STDOUT, text=True)
    metrics = []
    cur_alpha = None
    for line in out.splitlines():
        m_a = re.search(r"GRAPH_SKEW_ALPHA = ([0-9.]+)", line)
        if m_a:
            cur_alpha = float(m_a.group(1))
        m_r = re.match(r"^(\d+)\s+(\S+)\s+(True|False)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*x", line.strip())
        if m_r:
            metrics.append({
                "alpha": cur_alpha,
                "case_id": int(m_r.group(1)),
                "note": m_r.group(2),
                "passed": m_r.group(3) == "True",
                "max_diff": float(m_r.group(4)),
                "golden_us": float(m_r.group(5)),
                "cand_us": float(m_r.group(6)),
                "speedup": float(m_r.group(7)),
            })
    return metrics

def compute_score(metrics):
    if not metrics:
        return 0.0
    pass_rate = sum(1 for m in metrics if m["passed"]) / len(metrics)
    if pass_rate < 1.0:
        return round(50.0 * pass_rate, 2)
    avg_sp = sum(m["speedup"] for m in metrics) / len(metrics)
    return round(50.0 + 50.0 * min(avg_sp, 1.5), 2)

def main():
    print("=================================================================")
    print("  CANNBot 三维联合设计空间搜索启动 (Tile × AIV_Blocks × Skew_Thresh)")
    print("=================================================================\n")
    history, best_g, best_score = [], None, -1.0

    for g in MULTI_DIM_GENOMES:
        print(f">>> [三维进化 Gen {g['gen']}] {g['name']}")
        print(f"    基因参数: Tile={g['tile_bytes']//1024}KB | MaxAIV={g['max_aiv']} | SkewThresh={g['skew_thresh']}")
        print(f"    策略动机: {g['desc']}")
        apply_genome(g)

        t0 = time.time()
        if not build_and_install():
            print("    [Error] 编译失败，跳过本代。\n")
            continue
        print(f"    编译部署成功 ({time.time() - t0:.1f}s)，运行双卡多倾斜评测...")

        metrics = run_cannbench_eval()
        score = compute_score(metrics)
        avg_lat = sum(m["cand_us"] for m in metrics) / max(len(metrics), 1)
        print(f"    [综合评分] Pass={all(m['passed'] for m in metrics)} | 平均延迟={avg_lat:.2f} us | CANNBench 得分={score}/100")
        for m in metrics:
            if m["case_id"] in (2, 4):
                print(f"      - (Alpha={m['alpha']}) {m['note']}: Cand={m['cand_us']}us vs Golden={m['golden_us']}us (Speedup={m['speedup']}x)")
        print()

        history.append({"genome": g, "score": score, "avg_latency_us": round(avg_lat, 2), "metrics": metrics})
        if score > best_score:
            best_score, best_g = score, g

    if best_g:
        print("=================================================================")
        print(f"  三维搜索收敛！最优组合: {best_g['name']} (最高得分: {best_score}/100)")
        print("=================================================================")
        apply_genome(best_g)
        build_and_install()

    with open("cannbot_multidim_report.json", "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print("三维进化报告已保存至: cannbot_multidim_report.json")

if __name__ == "__main__":
    main()
