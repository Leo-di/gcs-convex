"""Fetch a pinned, metadata-only ReplicaCAD scene sample for Drake adaptation.

No simulator is installed and no mesh is downloaded. Original instance poses,
COM conventions and articulated-object definitions remain in the snapshots.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REPO = 'ai-habitat/ReplicaCAD_dataset'


def fetch(url):
    with urlopen(url, timeout=40) as response:
        return response.read()


def scout(output, scenes, revision=None):
    if revision is None:
        revision = json.loads(fetch(f'https://huggingface.co/api/datasets/{REPO}'))['sha']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('An exact 40-character revision SHA is required')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(repository=REPO, revision=revision, license='CC BY 4.0',
                    status='metadata_only_not_loaded_in_Drake', files={}, scenes=[])
    base = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/'
    def snapshot(path):
        local = output/path
        if not local.exists():
            data = fetch(base+path)
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            manifest['files'][path] = dict(sha256=hashlib.sha256(data).hexdigest(), url=base+path)
        return json.loads(local.read_text()) if path.endswith('.json') else local.read_text()
    snapshot('README.md')
    snapshot('replicaCAD.scene_dataset_config.json')
    for name in scenes:
        if not name.replace('_', '').isalnum():
            raise ValueError('Simple scene identifiers required')
        scene = snapshot(f'configs/scenes/{name}.scene_instance.json')
        stage = scene['stage_instance']['template_name']
        snapshot(f'configs/stages/{stage}.stage_config.json')
        objects = []
        for template in sorted({o['template_name'] for o in scene.get('object_instances', [])}):
            cfg = snapshot(f'configs/objects/{template}.object_config.json')
            objects.append(dict(template=template, render_asset=cfg.get('render_asset'),
                collision_asset=cfg.get('collision_asset'), mass=cfg.get('mass'),
                COM=cfg.get('COM'), join_collision_meshes=cfg.get('join_collision_meshes'),
                receptacles={k:v for k,v in cfg.get('user_defined', {}).items() if 'receptacle' in k}))
        manifest['scenes'].append(dict(id=name, stage=stage,
            rigid_instance_count=len(scene.get('object_instances', [])),
            articulated_instances=scene.get('articulated_object_instances', []), templates=objects))
    manifest['adaptation_requirements'] = [
        'Convert Y-up poses to Drake Z-up with an explicit rigid transform.',
        'Resolve COM versus asset origins, scale and quaternion conventions.',
        'Preserve convex collision components; do not replace the apartment with one convex hull.',
        'Import and initialize articulated URDF furniture; retain receptacle frames.',
        'Audit whole-robot and payload clearance; a base navmesh is insufficient.',
        'Define user constraints and independent task checks; asset reuse is not official benchmark evaluation.',
    ]
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(revision=revision, metadata_files=len(manifest['files']),
        scenes=[dict(id=s['id'], rigid_instances=s['rigid_instance_count'],
                     articulated_instances=len(s['articulated_instances'])) for s in manifest['scenes']]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', help='Exact dataset commit SHA; defaults to the current revision')
    parser.add_argument('--scenes', nargs='+', default=[
        'v3_sc0_staging_00', 'v3_sc1_staging_00', 'v3_sc2_staging_00'])
    args = parser.parse_args()
    scout(args.output, args.scenes, args.revision)
