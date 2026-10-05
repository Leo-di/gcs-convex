from src import online_candidate_repair as module


def test_retry_budget_preserves_failed_attempts_and_original_library(tmp_path,monkeypatch):
    seeds=[];library={'candidate':{'marker':[]}}
    def solve(disks,candidates,allowed,config,**kwargs):
        seeds.append(kwargs['rounding_seed']);candidates['candidate']['marker'].append(1)
        return {'status':'solver_failed','build_seconds':1.,'solve_seconds':2.},None
    monkeypatch.setattr(module,'build_and_solve',solve)
    result,audit,dest,history=module.solve_audited([],library,['candidate'],{},tmp_path,[0,0,0],7,10,'retry',3)
    assert seeds==[7,108,209,310]
    assert len(history)==4 and all(not h['accepted'] for h in history)
    assert library['candidate']['marker']==[]
    assert audit is None and result['solve_seconds']==8.


def test_first_audited_incumbent_stops_retries(tmp_path,monkeypatch):
    calls=[]
    def solve(*args,**kwargs):
        calls.append(kwargs['rounding_seed'])
        return {'status':'solved','build_seconds':0.,'solve_seconds':0.},[{}]
    monkeypatch.setattr(module,'build_and_solve',solve)
    monkeypatch.setattr(module,'expand',lambda *args,**kwargs:{'accepted':True})
    _,audit,_,history=module.solve_audited([],{},[],{},tmp_path,[0,0,0],19,10,'cuts',3)
    assert audit['accepted'] and calls==[19] and len(history)==1
