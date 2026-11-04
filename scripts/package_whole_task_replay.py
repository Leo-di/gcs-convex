"""Keep large recorded scene data external and expose task-time controls."""

import argparse
import json
import re
from pathlib import Path


def package(page, run):
    html = page.read_text()
    trace = json.loads((run / "physics.json").read_text())["trace"]
    duration = trace[-1]["time_s"]
    speed = json.loads(page.with_suffix(".json").read_text())["speed"]
    phases = []
    for row in trace:
        if not phases or phases[-1]["phase"] != row["phase"]:
            phases.append(
                dict(phase=row["phase"], start_s=row["time_s"], end_s=row["time_s"])
            )
        else:
            phases[-1]["end_s"] = row["time_s"]
    count = 0

    def external(match):
        nonlocal count
        content = match.group(1)
        if len(content) < 100000:
            return match.group(0)
        count += 1
        target = page.with_suffix(f".data{count}.js")
        target.write_text(content)
        return f'<script src="{target.name}"></script>'

    html = re.sub(r"<script[^>]*>(.*?)</script>", external, html, flags=re.S)
    html = html.replace(
        "Functional access constraint", "One whole-task joint solve · 3 cups"
    )
    html = html.replace("Playback speed</span>", "Playback multiplier</span>")
    controls = (
        """<style>.lil-gui,.dg,.guify{display:none!important}</style>
<div style="position:fixed;bottom:85px;left:20px;right:20px;background:#fffffff0;padding:10px;font:16px Arial;z-index:10001">
<button id="taskPlay">Play</button> <button id="taskPause">Pause</button>
<label>Simulation time <input id="taskTime" type="range" min="0" max="DURATION" step="0.05" value="0" style="width:50%" aria-label="Full task simulation time"></label> <span id="taskPhase"></span></div>
<script>
const taskPhases=PHASES;const timeSlider=document.getElementById('taskTime');
document.getElementById('taskPlay').onclick=()=>viewer.animator.play();
document.getElementById('taskPause').onclick=()=>viewer.animator.pause();
timeSlider.oninput=()=>{viewer.animator.pause();let t=Number(timeSlider.value)/SPEED;viewer.animator.seek(t);viewer.animator.display_progress(t);};
setInterval(()=>{if(typeof viewer==='undefined'||!viewer.animator.actions.length)return;if(viewer.gui)viewer.gui.domElement.style.display='none';let t=Math.min(Number(timeSlider.max),viewer.animator.time*SPEED+.01);if(document.activeElement!==timeSlider)timeSlider.value=t;let p=taskPhases.find(p=>t>=p.start_s&&t<=p.end_s+.05);document.getElementById('taskPhase').textContent=t.toFixed(1)+' s '+(p?p.phase:'');},150);
</script>""".replace("DURATION", str(duration))
        .replace("PHASES", json.dumps(phases))
        .replace("SPEED", str(speed))
    )
    html = html.replace("</body>", controls + "</body>")
    page.write_text(html)
    print("Packaged", page, "external assets", count)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--page", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    package(a.page, a.run)
