"""analyze.py - reads results/*.json and produces tables (results/summary.md) and plots (results/*.png)."""
import glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob("results/*.json")}
def mean_epoch(r): return sum(e["time"] for e in r["epochs"][1:]) / max(1, len(r["epochs"]) - 1)  # skip epoch 1 (cudnn warm-up)
def final_acc(r): return sum(e["test_acc"] for e in r["epochs"][-3:]) / 3
def tta(r, target):
    for e in r["epochs"]:
        if e["test_acc"] >= target: return e["cum_time"]
    return None

lines = []
for mode in ["weak", "strong"]:
    runs = sorted([(r["world"], r) for k, r in R.items() if k.startswith(mode + "_N")], key=lambda t: t[0])
    if not runs: continue
    t1 = mean_epoch(runs[0][1]); a1 = final_acc(runs[0][1])
    lines += [f"\n### {mode.capitalize()} scaling\n",
              "| N | global batch | epoch time (s) | speedup | efficiency (%) | total time (s) | final acc (%) | acc gap | comm/step (ms) | exposed comm (% of step) | raw all-reduce (ms) | data wait (% epoch) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    Ns, S, E = [], [], []
    for n, r in runs:
        te = mean_epoch(r)
        # weak scaling: work per epoch is constant (dataset fixed) so speedup = T1/Tn in time per epoch
        s = t1 / te; eff = 100 * s / n
        exp = r["t_full_step"] - r["t_compute_step"]
        dw = 100 * sum(e["data_wait"] for e in r["epochs"][1:]) / sum(e["time"] for e in r["epochs"][1:])
        lines.append(f"| {n} | {r['global_bs']} | {te:.2f} | {s:.2f} | {eff:.1f} | {sum(e['time'] for e in r['epochs']):.0f} | "
                     f"{final_acc(r):.2f} | {a1 - final_acc(r):+.2f} | {exp*1e3:.1f} | {100*max(exp,0)/r['t_full_step']:.1f} | {r['t_allreduce_raw']*1e3:.1f} | {dw:.1f} |")
        Ns.append(n); S.append(s); E.append(eff)
    plt.figure(figsize=(9, 3.5))
    plt.subplot(1, 2, 1); plt.plot(Ns, S, "o-", label="measured"); plt.plot(Ns, Ns, "k--", label="ideal")
    plt.xlabel("workers (GPUs)"); plt.ylabel("speedup S(N)"); plt.title(f"{mode} scaling: speedup"); plt.legend(); plt.grid(alpha=.3)
    plt.subplot(1, 2, 2); plt.plot(Ns, E, "s-r"); plt.ylim(0, 105)
    plt.xlabel("workers (GPUs)"); plt.ylabel("efficiency (%)"); plt.title("parallel efficiency"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"results/{mode}_speedup.png", dpi=160); plt.close()

# convergence curves: accuracy vs wall-clock time and vs epoch
for mode in ["weak", "strong"]:
    runs = sorted([(r["world"], r) for k, r in R.items() if k.startswith(mode + "_N")], key=lambda t: t[0])
    if not runs: continue
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for n, r in runs:
        ax[0].plot([e["cum_time"] for e in r["epochs"]], [e["test_acc"] for e in r["epochs"]], label=f"N={n}")
        ax[1].plot([e["epoch"] for e in r["epochs"]], [e["test_acc"] for e in r["epochs"]], label=f"N={n}")
    ax[0].set_xlabel("training time (s)"); ax[1].set_xlabel("epoch")
    for a in ax: a.set_ylabel("test accuracy (%)"); a.set_ylim(50, 100); a.grid(alpha=.3); a.legend()
    ax[0].set_title(f"{mode}: accuracy vs time"); ax[1].set_title(f"{mode}: accuracy vs epoch")
    plt.tight_layout(); plt.savefig(f"results/{mode}_convergence.png", dpi=160); plt.close()

# time-to-accuracy table (90 %)
lines += ["\n### Time to 90% test accuracy (s)\n", "| run | time (s) |", "|---|---|"]
for k in sorted(R): lines.append(f"| {k} | {tta(R[k], 90) if tta(R[k], 90) else 'not reached'} |")
open("results/summary.md", "w").write("\n".join(lines)); print("\n".join(lines))
