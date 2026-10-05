#!/usr/bin/env bash
# 一键查看 SE(2) 回归补跑状态：./status_regression.sh
cd "$(dirname "$0")"
echo "=== $(date '+%H:%M:%S')  load:$(cut -d, -f1 /proc/loadavg) ==="
echo "--- 补跑日志 ---"
grep -E 'PATCH_(START|DONE|COMPLETE)|PATCH_SCENES' /tmp/se2_patch2.log 2>/dev/null | tail -8
echo "--- 各场景候选状态（旧版候选数 | 新版进度）---"
for s in $(ls outputs/results/se2_navigation_refresh_v3/libraries/ | sort); do
  old=$(cat outputs/results/heldout_transfer_v1/raw/libraries/$s/preparation.json 2>/dev/null | .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["candidate_count"])' 2>/dev/null)
  f=outputs/results/se2_navigation_refresh_v3/libraries/$s/preparation_progress.json
  g=outputs/results/se2_navigation_refresh_v3/libraries/$s/preparation.json
  if [ -f "$f" ]; then
    new=$(cat $f | .venv/bin/python -c 'import json,sys; d=json.load(sys.stdin); print(", ".join(a["name"]+"="+("OK" if a["status"]=="accepted" else "X") for a in d["attempts"]))' 2>/dev/null)
    echo "$s (旧=$old): $new [进行中]"
  elif [ -f "$g" ]; then
    cnt=$(cat $g | .venv/bin/python -c 'import json,sys; d=json.load(sys.stdin); print(d.get("candidate_count","?"))' 2>/dev/null)
    fail=$(cat $g | .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin).get("failure") or "")' 2>/dev/null)
    echo "$s (旧=$old): 新=$cnt $fail [完成]"
  fi
done
echo "--- 存活 worker ---"
ps aux | grep -c '[n]avigation_refresh'
