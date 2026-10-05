"""Check all successful and failed pilot display conditions in Chromium."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args();summary=json.loads((args.source/'summary.json').read_text())
    url='http://127.0.0.1:7001/online_iris/';errors=[];checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox','--enable-webgl','--use-gl=angle','--use-angle=swiftshader'])
        page=browser.new_page(viewport={'width':1440,'height':1000})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(url,wait_until='networkidle');page.wait_for_function("document.querySelectorAll('#metrics tr').length===6")
        for query in summary['protocol']['queries']:
            page.select_option('#query',str(query))
            has_success=any(r['online']['accepted'] for r in summary['rows'] if r['query']==query)
            if has_success:
                page.wait_for_function("q => document.querySelector('#ready').textContent==='记录已就绪' && document.querySelector('#scene').contentWindow.location.pathname.endsWith('/case_'+String(q).padStart(3,'0')+'/scene.html')",arg=query,timeout=60000)
            else:page.wait_for_function("document.querySelector('#ready').textContent.includes('两组均无')")
            for method in ['uniform','vcc']:
                row=next(r for r in summary['rows'] if r['query']==query and r['method']==method)
                page.select_option('#method',method)
                if row['online']['accepted']:
                    assert page.locator('#play').is_enabled()
                    page.locator('#play').click();page.wait_for_function("+document.querySelector('#seek').value>=5");page.locator('#play').click()
                    page.locator('#seek').evaluate("e=>{e.value=150;e.dispatchEvent(new Event('input'));}")
                    observed=page.evaluate("document.querySelector('#scene').contentWindow.viewer.animator.time")
                    expected=page.evaluate("data.recordings[method.value][150].time")
                    assert abs(observed-expected)<1e-6
                    assert page.locator('#scene').is_visible()
                else:
                    assert page.locator('#play').is_disabled()
                    assert '无通过审核' in page.locator('#state').inner_text()
                    assert not page.locator('#scene').is_visible()
                checks.append(dict(query=query,method=method,accepted=row['online']['accepted'],display_checked=True))
            if query==1:
                page.locator('#state').scroll_into_view_if_needed();page.screenshot(path=str(args.source/'online_iris_replay.png'))
        assert not errors,errors
        result=dict(url=url,rows=6,checks=checks,page_errors=errors)
        (args.source/'browser_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));browser.close()

if __name__=='__main__':main()
