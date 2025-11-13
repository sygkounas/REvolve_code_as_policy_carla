# hv_finalize.py — SMS-EMOA finalize (RAW-F1 fitness + last-front deletion)
# - Objectives: [lane, progress, smooth, safety] in [0,1], maximize
# - REF = -1e-3 (strictly worse than all points) -> avoids HV=0 degeneracy
# - Writes RAW ΔHV on F1 (Pareto set) into fitness_scores/{gen}_{ctr}.txt ; dominated -> 0
# - Deletion (only if over cap): smallest ΔHV on LAST front (SMS-EMOA rule)
# - Prints a "sampling preview" = softmax probs per island at T in {0.5, 1.0, 2.0}

import os, json, glob, math
from itertools import product

# -------- CONFIG --------
REWARDS_DIR = ".../database/revolve_auto/1"
MAX_ISLAND_SIZE = int(os.environ.get("MAX_ISLAND_SIZE", "10"))
REF = -1e-3     # keep constant across runs
TEMPS = [0.01, 0.05, 0.1]  # preview temperatures for sampling
# ------------------------

def info(msg): print(msg, flush=True)

def read_json(p):
    with open(p, "r") as f:
        return json.load(f)

def write_float(p, x):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(f"{float(x):.6f}\n")

def parse_gen_ctr(path_or_name):
    stem = os.path.splitext(os.path.basename(path_or_name))[0]
    g, c = stem.split("_")
    return int(g), int(c)

def list_islands(root):
    return sorted(d for d in glob.glob(os.path.join(root, "island_*")) if os.path.isdir(d))

def load_population(island_dir):
    pop = []
    jfiles = sorted(glob.glob(os.path.join(island_dir, "policy_history", "*.json")))
    for jpath in jfiles:
        data = read_json(jpath)
        metrics = data.get("metrics", {})

        def mean(key):
            arr = metrics.get(key, [])
            return float(sum(arr)/len(arr)) if arr else 0.0

        lane     = mean("lane_score")
        progress = mean("progress")
        smooth   = mean("smooth_score")

        col = metrics.get("collision", [])
        red = metrics.get("num_red_violations", [])
        n   = max(len(col), len(red), 1)
        col = list(map(float, col)) + [0.0]*(n-len(col))
        red = list(map(float, red)) + [0.0]*(n-len(red))
        safety = sum(1.0 if (col[i]==0.0 and red[i]==0.0) else 0.0 for i in range(n)) / n
        efficiency = mean("eff_score")  # 0–1 range
       # obj = [lane, progress, smooth, safety, efficiency]
        obj = [lane, progress, smooth, safety]
        gen, ctr = parse_gen_ctr(jpath)
        pop.append({
            "gen": gen, "ctr": ctr, "obj": obj,
            "json": jpath,
            "policy": os.path.join(island_dir, "policies", f"{gen}_{ctr}.txt"),
            "fitfile": os.path.join(island_dir, "fitness_scores", f"{gen}_{ctr}.txt"),
            "oldfit": os.path.join(island_dir, "old_fit", f"{gen}_{ctr}.txt"),
        })
    return pop

def dominates(a, b):
    ge = all(x >= y for x,y in zip(a,b))
    gt = any(x > y for x,y in zip(a,b))
    return ge and gt

def nondominated_sort(points):
    n = len(points)
    S = [[] for _ in range(n)]; n_dom = [0]*n; fronts=[]
    for i in range(n):
        for j in range(n):
            if i==j: continue
            if dominates(points[i], points[j]): S[i].append(j)
            elif dominates(points[j], points[i]): n_dom[i]+=1
    F = [i for i in range(n) if n_dom[i]==0]
    rank = [-1]*n
    cur = 1
    while F:
        for i in F: rank[i] = cur
        fronts.append(F)
        nxt=[]
        for i in F:
            for j in S[i]:
                n_dom[j]-=1
                if n_dom[j]==0: nxt.append(j)
        F = nxt
        cur += 1
    return fronts, rank

# ---- HV contribution on an arbitrary subset of indices (e.g., F1) ----
def hv_contrib_on_indices(points, idx_list, ref):
    n = len(points)
    if not idx_list: return [0.0]*n, 0.0
    M = len(points[0])
    sub = [points[i] for i in idx_list]

    axes = []
    for d in range(M):
        coords = sorted(set([ref] + [p[d] for p in sub]))
        axes.append(coords)

    total = 0.0
    contrib_local = [0.0]*len(sub)
    for edges in product(*[list(zip(ax[:-1], ax[1:])) for ax in axes]):
        vol = 1.0
        for a,b in edges:
            dv = b - a
            if dv <= 0.0: vol = 0.0; break
            vol *= dv
        if vol == 0.0: continue
        ups = [b for (a,b) in edges]
        covered = [j for j,p in enumerate(sub) if all(ups[d] <= p[d] for d in range(M))]
        if covered:
            total += vol
            if len(covered) == 1:
                contrib_local[covered[0]] += vol

    # scatter back into full-length contrib (RAW, not normalized)
    contrib_full = [0.0]*n
    for pos, idx in enumerate(idx_list):
        contrib_full[idx] = contrib_local[pos]
    return contrib_full, total

# ---- LAST front HV contrib (for SMS-EMOA deletion) ----
def hv_contrib_last_front(points, ref):
    n = len(points)
    if n == 0: return [0.0]*n, 0.0, []
    fronts, _ = nondominated_sort(points)
    Fk = fronts[-1] if fronts else []
    if not Fk: return [0.0]*n, 0.0, []

    # reuse generic routine
    contrib_raw, total = hv_contrib_on_indices(points, Fk, ref)
    return contrib_raw, total, Fk

def remove_files(ind):
    for p, tag in ((ind["fitfile"], "fitness"), (ind["json"], "json"),
                   (ind["policy"], "policy"), ):
        try:
            if os.path.exists(p):
                os.remove(p)
                info(f"      [DEL] {tag}: {p}")
        except Exception as e:
            info(f"      [DEL] WARN cannot delete {tag} {p}: {e}")

def sms_emoa_trim(pop, cap, ref):
    # delete smallest ΔHV in LAST front until size <= cap
    while len(pop) > cap:
        points = [x["obj"] for x in pop]
        contrib_raw, total, Fk = hv_contrib_last_front(points, ref)
        if not Fk:
            worst = min(range(len(pop)), key=lambda i: sum(pop[i]["obj"]))
            info("    [TRIM] degenerate F: removing min sum(obj)")
        else:
            minc = min(contrib_raw[i] for i in Fk)
            cands = [i for i in Fk if abs(contrib_raw[i]-minc) <= 1e-18]
            worst = min(cands, key=lambda i: (pop[i]["gen"], pop[i]["ctr"]))
            info(f"    [TRIM] F_last size={len(Fk)} min ΔHV(raw)={minc:.6e} candidates={[(pop[i]['gen'],pop[i]['ctr']) for i in cands]}")
        ind = pop[worst]
        info(f"    [TRIM] remove gen={ind['gen']} ctr={ind['ctr']} obj={[round(x,6) for x in ind['obj']]}")
        remove_files(ind)
        pop.pop(worst)

def softmax(scores, T=1.0):
    # stable softmax
    m = max(scores) if scores else 0.0
    exps = [math.exp((s - m)/T) for s in scores]
    Z = sum(exps) if exps else 1.0
    return [e / Z for e in exps]

def read_fitness_scores(island_dir):
    vals = {}
    for f in sorted(glob.glob(os.path.join(island_dir, "fitness_scores", "*.txt"))):
        g,c = parse_gen_ctr(f)
        try:
            with open(f, "r") as fh:
                x = float(fh.read().strip())
        except Exception:
            x = 0.0
        vals[(g,c)] = x
    return vals

def main():
    info(f"[hv_finalize] REWARDS_DIR={REWARDS_DIR}  MAX_ISLAND_SIZE={MAX_ISLAND_SIZE}  REF={REF}")
    isl = list_islands(REWARDS_DIR)
    if not isl:
        info("[hv_finalize] No islands found.")
        return

    for island_dir in isl:
        info(f"\n[hv_finalize] Island: {island_dir}")
        pop = load_population(island_dir)
        info(f"  individuals found: {len(pop)}")
        if not pop: continue

        # print objectives
        for ind in pop:
            info(f"    IND gen={ind['gen']} ctr={ind['ctr']} obj={[round(x,6) for x in ind['obj']]}")
        points = [x["obj"] for x in pop]
        M = len(points[0])
        mins = [min(p[d] for p in points) for d in range(M)]
        maxs = [max(p[d] for p in points) for d in range(M)]
        info(f"  per-axis min = {[round(x,6) for x in mins]}")
        info(f"  per-axis max = {[round(x,6) for x in maxs]}")

        fronts, rank = nondominated_sort(points)
        info(f"  #fronts={len(fronts)}  front sizes={[len(f) for f in fronts]}")
        for r, F in enumerate(fronts, start=1):
            members = [(pop[i]["gen"], pop[i]["ctr"]) for i in F]
            info(f"    F{r}: {members}")

        # ----- FITNESS TO WRITE: RAW ΔHV on F1 (Pareto set) -----
        F1 = fronts[0] if fronts else []
        cF1_raw, HV_F1 = hv_contrib_on_indices(points, F1, REF)
        info(f"  HV(F1) = {HV_F1:.6e}  (RAW ΔHV written to fitness_scores; dominated -> 0)")

        # write RAW ΔHV(F1) as fitness
        for i, ind in enumerate(pop):
            write_float(ind["fitfile"], cF1_raw[i])

            # --- also write HV inside metrics of policy_history JSON ---
            json_path = ind["json"]
            hv_value = float(cF1_raw[i])

            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                except Exception:
                    data = {"metrics": {}}
            else:
                data = {"metrics": {}}

            if "metrics" not in data:
                data["metrics"] = {}
            data["metrics"]["fitness"] = hv_value
            data["fitness"] = hv_value

            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

            info(f"    [JSON] Updated {json_path} → fitness={hv_value:.6e}")

        # preview: show RAW ΔHV per individual
        for i, ind in enumerate(pop):
            mark = "*" if i in F1 else " "
            info(f"    {mark} F1-ΔHV(raw) gen={ind['gen']} ctr={ind['ctr']}  val={cF1_raw[i]:.6e}")

        # ----- SMS-EMOA DELETION (ONLY IF OVER CAP): LAST FRONT ΔHV -----
        if len(pop) > MAX_ISLAND_SIZE:
            info(f"  trimming {len(pop)} → {MAX_ISLAND_SIZE} via SMS-EMOA (last-front smallest ΔHV raw)...")
            sms_emoa_trim(pop, MAX_ISLAND_SIZE, REF)
            info("  trimming complete")
        else:
            # still compute/report last-front HV for debug
            contrib_last_raw, HV_last, Fk = hv_contrib_last_front(points, REF)
            info(f"  [DBG] LAST-FRONT size = {len(Fk)} | HV(F_last) = {HV_last:.6e}")
            if Fk:
                for i in Fk:
                    ind = pop[i]
                    info(f"      LAST-F ΔHV(raw) gen={ind['gen']} ctr={ind['ctr']} = {contrib_last_raw[i]:.6e}")
            info(f"  no trimming needed (size {len(pop)} ≤ cap {MAX_ISLAND_SIZE})")

        # ----- SAMPLING PREVIEW (softmax over the fitness we just wrote) -----
        scores = read_fitness_scores(island_dir)   # {(gen,cnt):score}
        keys_sorted = sorted(scores.keys())        # stable order
        vals = [scores[k] for k in keys_sorted]
        mean_fit = sum(vals)/len(vals) if vals else 0.0
        info(f"  [PREVIEW] island mean fitness (RAW-F1) = {mean_fit:.6e}")
        for T in TEMPS:
            probs = softmax(vals, T=T)
            info(f"    softmax T={T:.2f}:")
            for (g,c), s, p in zip(keys_sorted, vals, probs):
                info(f"      ({g},{c}) score={s:.6e}  p={p:.4f}")

    info("\n[hv_finalize] DONE")

if __name__ == "__main__":
    main()
