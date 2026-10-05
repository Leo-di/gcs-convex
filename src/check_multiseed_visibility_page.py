"""Exercise all seed/method replays in the published local dashboard."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7001/visibility_clique_multiseed/')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    checks = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=[
            '--no-sandbox', '--enable-webgl', '--use-gl=angle', '--use-angle=swiftshader'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until='networkidle')
        page.wait_for_function("document.querySelectorAll('#metrics tr').length === 3")
        for seed in ['1701', '1702', '1703']:
            page.select_option('#seed', seed)
            page.wait_for_function("s => document.querySelector('#display').textContent.includes('种子 '+s) && document.querySelector('#ready').textContent === '三维记录已就绪'", arg=seed, timeout=60000)
            for method in ['uniform', 'vcc']:
                page.select_option('#method', method)
                assert page.locator('#play').is_enabled()
                page.locator('#play').click()
                page.wait_for_function("+document.querySelector('#seek').value >= 5")
                page.locator('#play').click()
                page.locator('#seek').evaluate("e => {e.value = 150; e.dispatchEvent(new Event('input'));}")
                assert '七关节构型' in page.locator('#state').inner_text()
                info = page.evaluate("""() => {
                    const w=document.querySelector('#scene').contentWindow;
                    return {animator_time:w.viewer.animator.time, canvas_count:w.document.querySelectorAll('canvas').length,
                            slider:Number(document.querySelector('#seek').value), scene_url:w.location.pathname};
                }""")
                assert info['slider'] == 150 and info['canvas_count'] > 0
                expected = page.evaluate("data.recordings[method.value][150].time")
                assert abs(info['animator_time'] - expected) < 1e-6
                checks.append(dict(seed=seed, method=method, **info))
            if seed == '1701':
                page.locator('#scene').scroll_into_view_if_needed()
                page.screenshot(path=str(args.output / 'multiseed_replay.png'))
        assert not errors, errors
        result = dict(url=args.url, table_rows=3, replay_checks=checks, page_errors=errors)
        (args.output / 'browser_verification.json').write_text(json.dumps(result, indent=2))
        browser.close()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
