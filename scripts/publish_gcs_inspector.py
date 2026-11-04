"""Render saved GCS evidence without solving again or changing frozen artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def projected_envelope(A, b, axes=(0, 3), directions=32):
    """Outer approximation of a 2D projection using LP support halfplanes.

    All unshown coordinates remain free. This is not a fixed-coordinate slice,
    nor a certified free-space region. Floating-point LP tolerance applies.
    """
    A, b = np.asarray(A, float), np.asarray(b, float)
    if directions < 4 or directions % 4 or len(set(axes)) != 2:
        raise ValueError('Distinct axes and a multiple of four support directions required')
    normals, supports = [], []
    for angle in np.linspace(0, 2 * np.pi, directions, endpoint=False):
        n = np.array([np.cos(angle), np.sin(angle)])
        c = np.zeros(A.shape[1]); c[list(axes)] = -n
        result = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * len(c), method='highs')
        if not result.success:
            raise ValueError(f'Cannot project polytope: {result.message}')
        normals.append(n); supports.append(float(-result.fun + 1e-9))
    lo = [-supports[directions // 2], -supports[3 * directions // 4]]
    hi = [supports[0], supports[directions // 4]]
    polygon = [np.array(p) for p in [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]]
    for n, h in zip(normals, supports):
        clipped = []
        for a, z in zip(polygon, polygon[1:] + polygon[:1]):
            da, dz = float(n @ a - h), float(n @ z - h)
            if da <= 0:
                clipped.append(a)
            if (da <= 0) != (dz <= 0):
                clipped.append(a + da / (da - dz) * (z - a))
        polygon = clipped
    if len(polygon) < 3:
        raise ValueError('Degenerate projected envelope')
    return np.asarray(polygon).tolist()


def publish(candidate, output):
    from src.meshcat_playback import add_playback_controls
    candidate, output = Path(candidate).resolve(), Path(output).resolve()
    if output == candidate or candidate in output.parents:
        raise ValueError('Write the viewer outside the immutable experiment candidate')
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def read(name):
        raw = (candidate / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def url(name):
        return '/' + (candidate / name).relative_to(PROJECT).as_posix()

    config = read('scene_config_snapshot.json')
    result = read('candidate_result.json')
    if not result.get('composition', {}).get('accepted') or result.get('stage') != 'complete':
        raise ValueError('Viewer requires a completed accepted candidate')
    payload = {'candidate': candidate.relative_to(PROJECT).as_posix(),
               'config': config, 'physical': result['physical'], 'stages': {}}
    for phase in ('reach', 'lift'):
        cover = read(f'{phase}_cover.json')
        path = read(f'{phase}_cover_path.json')
        regions = []
        for i, region in enumerate(cover['regions']):
            regions.append({'id': i, 'polygon': projected_envelope(region['A'], region['b']),
                            'facets': len(region['b']),
                            'room_bound_m': region['whole_box_semantic_lower_bound_m']})
        payload['stages'][phase] = dict(regions=regions, edges=cover['edges'],
            sources=cover['sources'], targets=cover['targets'], axes=[0, 3],
            raw=path['gcs_path'], execution=path['execution_path'],
            cover_url=url(f'{phase}_cover.json'), path_url=url(f'{phase}_cover_path.json'))
        print(f'{phase}: {len(regions)} regions projected', flush=True)
    nav = read('rotation_safe_cover.json')
    if not np.allclose(nav['scale'][:2], [1, 1]):
        raise ValueError('XY disk rendering requires unit translation scales')
    if any(not nav['domain_lower'][2] <= r['center'][2] <= nav['domain_upper'][2]
           for r in nav['regions']):
        raise ValueError('XY disk rendering requires centers inside the yaw domain')
    nav['raw'] = read('bridge_events.json')['path']
    nav['execution'] = [x['base_q'] for x in read('delivery_trace.json') if x['phase'] == 'navigate_to_pickup']
    nav['cover_url'] = url('rotation_safe_cover.json')
    nav['path_url'] = url('bridge_events.json')
    payload['navigation'] = nav
    payload['projection_note'] = '32 LP support directions; 2D outer approximation of saved 7D Aq<=b, not a slice or feasibility certificate.'
    replay = candidate / 'execution/execution_grounded_candidate/scene.html'
    hashes[str(replay.relative_to(candidate))] = hashlib.sha256(replay.read_bytes()).hexdigest()
    (output / 'replay.html').write_text(add_playback_controls(replay.read_text()))
    (output / 'data.json').write_text(json.dumps(payload, ensure_ascii=False))
    template = (PROJECT / 'scripts/gcs_inspector.html').read_text()
    (output / 'index.html').write_text(template)
    (output / 'provenance.json').write_text(json.dumps(dict(candidate=str(candidate), sha256=hashes,
        scope='Visualization only. No replanning. Frozen source artifacts unchanged.'), indent=2))
    print(output / 'index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    publish(args.candidate, args.output)
