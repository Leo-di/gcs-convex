"""Static layout inspection; separate from executed trajectory animations."""
import argparse,json
from pathlib import Path
from .scene_spec import load_scene_config
from .build_scene import build_scene


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);out=root/'layouts';out.mkdir(exist_ok=True)
    camera={'eye':[1.8,1.3,2.7],'target':[.65,.65,.95]};(out/'camera.json').write_text(json.dumps(camera,indent=2))
    for task in json.loads((root/'protocol.json').read_text())['tasks']:
        name=task['id'];scene=build_scene(config=load_scene_config(root/'configs'/f'{name}.json'),visualize=True);scene.meshcat.SetCameraPose(camera['eye'],camera['target']);scene.diagram.ForcedPublish(scene.root_context);(out/f'{name}.html').write_text(scene.meshcat.StaticHtml())

if __name__=='__main__':main()
