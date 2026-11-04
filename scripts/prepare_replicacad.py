"""Convert a pinned ReplicaCAD scene and reusable cafe furniture for Drake."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from src.replicacad import prepare_stage, prepare_furniture, urdf_snapshot, transform, Y_UP_TO_Z_UP, REVISION


def prepare(root,output,scene_id):
    root=root.resolve();output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    source=json.loads((root/f'configs/scenes/{scene_id}.scene_instance.json').read_text())
    C=np.eye(4);C[:3,:3]=Y_UP_TO_Z_UP
    entries=[]
    stage=source['stage_instance']
    stage_dir=output/'architecture'
    asset=(json.loads((stage_dir/'asset.json').read_text()) if (stage_dir/'asset.json').exists()
           else prepare_stage(root/f'stages/{stage["template_name"]}.glb',stage_dir))
    entries.append(dict(name='replicacad_architecture',asset=asset,pose=(C @ transform(stage['translation'],stage['rotation']) @ np.linalg.inv(C)).tolist()))
    for i,instance in enumerate(source.get('object_instances',[])):
        name=instance['template_name'];folder=output/'furniture'/name
        asset=(json.loads((folder/'asset.json').read_text()) if (folder/'asset.json').exists()
               else prepare_furniture(root,name,folder))
        X=C @ transform(instance['translation'],instance['rotation']) @ np.linalg.inv(C)
        entries.append(dict(name=f'rc_{i:02d}_{name}',asset=asset,pose=X.tolist()))
    for i,instance in enumerate(source.get('articulated_object_instances',[])):
        folder=output/'articulated'/f'{i:02d}_{instance["template_name"]}'
        asset=urdf_snapshot(root,instance,folder)
        entries.append(dict(name=f'rc_joint_{i:02d}_{instance["template_name"]}',asset=asset,
            pose=(C @ transform(instance['translation'],instance['rotation'])).tolist()))
    manifest=dict(scene=scene_id,source_revision=REVISION,license='CC BY 4.0',
        source_root=str(root),entries=entries,
        scope='Rigid furniture and URDF snapshots; architecture approximation recorded per component; original cutaway visuals')
    (output/'scene.json').write_text(json.dumps(manifest,indent=2))
    # Cafe uses the same source table/chair meshes, with a stated geometric
    # normalization. Body centers are at the center of each requested box.
    for name,size in [('frl_apartment_table_01',[1.2,.8,.75]),
                      ('frl_apartment_chair_04',[.48,.48,.88])]:
        prepare_furniture(root,name,output/'cafe'/name,size=size)
    print('Prepared',len(entries),'fixed asset bodies,',sum(len(e['asset']['collisions']) for e in entries),'collision components',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=PROJECT/'data/replicacad')
    p.add_argument('--output',type=Path,default=PROJECT/'data/replicacad_drake')
    p.add_argument('--scene',default='v3_sc0_staging_00')
    a=p.parse_args();prepare(a.assets,a.output,a.scene)
