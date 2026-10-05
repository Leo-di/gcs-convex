"""Check the complete table and all replay records, plus representative WebGL playback."""
import argparse
import json
from pathlib import Path
import urllib.request
from playwright.sync_api import sync_playwright


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    root=p.parse_args().source;summary=json.loads((root/'summary.json').read_text());url='http://127.0.0.1:7001/online_iris_100q/'
    # Check all 200 data conditions, including the absence of frames on failure.
    for q in summary['protocol']['queries']:
        with urllib.request.urlopen(url+f'case_{q:03}/data.json') as response:data=json.load(response)
        assert data['display_query_index']==q
        for m in ['uniform','vcc']:
            row=next(r for r in summary['rows'] if r['query']==q and r['method']==m)
            assert len(data['recordings'][m])==(301 if row['online']['accepted'] else 0)
    failures=[r['query'] for r in summary['rows'] if not r['online']['accepted']]
    both_failed=[q for q in summary['protocol']['queries'] if all(not r['online']['accepted'] for r in summary['rows'] if r['query']==q)]
    audit_failed=[r['query'] for r in summary['rows'] if r['online']['failure_category']=='trajectory_audit_failed']
    queries=sorted(set([1,2,3,100]+failures[:1]+both_failed[:1]+audit_failed[:1]));errors=[];checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox','--enable-webgl','--use-gl=angle','--use-angle=swiftshader'])
        page=browser.new_page(viewport={'width':1440,'height':1000});page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(url,wait_until='networkidle');page.wait_for_function("document.querySelectorAll('#metrics tr').length===200 && document.querySelectorAll('#aggregate tr').length===2")
        assert page.locator('#query option').count()==100
        page.screenshot(path=str(root/'batch_metrics.png'))
        for q in queries:
            page.select_option('#query',str(q));success=any(r['query']==q and r['online']['accepted'] for r in summary['rows'])
            if success:page.wait_for_function("q=>document.querySelector('#ready').textContent==='记录已就绪' && document.querySelector('#scene').contentWindow.location.pathname.endsWith('/case_'+String(q).padStart(3,'0')+'/scene.html')",arg=q,timeout=60000)
            else:page.wait_for_function("document.querySelector('#ready').textContent.includes('两组均无')")
            for m in ['uniform','vcc']:
                row=next(r for r in summary['rows'] if r['query']==q and r['method']==m);page.select_option('#method',m)
                if row['online']['accepted']:
                    assert page.locator('#play').is_enabled();page.locator('#play').click();page.wait_for_function("+document.querySelector('#seek').value>=5");page.locator('#play').click()
                    page.locator('#seek').evaluate("e=>{e.value=150;e.dispatchEvent(new Event('input'));}")
                    assert abs(page.evaluate("document.querySelector('#scene').contentWindow.viewer.animator.time")-page.evaluate("data.recordings[method.value][150].time"))<1e-6
                else:
                    assert page.locator('#play').is_disabled() and not page.locator('#scene').is_visible()
                    assert '无通过审核' in page.locator('#state').inner_text()
                checks.append(dict(query=q,method=m,accepted=row['online']['accepted']))
            if q==1:page.locator('#scene').scroll_into_view_if_needed();page.screenshot(path=str(root/'batch_replay.png'))
        assert not errors,errors
        result=dict(url=url,table_rows=200,query_options=100,replay_data_conditions_checked=200,interactive_checks=checks,page_errors=errors)
        (root/'browser_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));browser.close()

if __name__=='__main__':main()
