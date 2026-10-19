"""Stop only at a complete, physically verified grounding; preserve attempts."""
import json,time
from pathlib import Path


def search_groundings(candidates,evaluate,budget_seconds,output):
    began=time.monotonic();attempts=[];result={'selected':None,'success':False,'attempts':attempts}
    for candidate in candidates:
        remaining=budget_seconds-(time.monotonic()-began)
        if remaining<=0:break
        # Evaluator must enforce the remaining wall-time budget in its process.
        row=evaluate(candidate,remaining)
        complete=bool(row.get('process_exit')==0 and all(row.get(k,{}).get('accepted') for k in ['bridge','operation','physical']))
        attempts.append({'id':candidate['id'],'complete':complete,'failure_stage':row.get('failure_stage',row.get('stage')),'failure_reason':row.get('failure_reason',row.get('bridge',{}).get('reason'))})
        if complete:result.update(selected=candidate['id'],success=True)
        result['elapsed_seconds']=time.monotonic()-began
        Path(output).write_text(json.dumps(result,indent=2))
        if complete:return result
    result['elapsed_seconds']=time.monotonic()-began;result['termination']='budget_or_candidates_exhausted'
    Path(output).write_text(json.dumps(result,indent=2));return result
