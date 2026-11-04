"""Train/load a geometric room surrogate with a mandatory analytical guard.

Labels are robot geometry, not human preference annotations. Disjoint seeded
train/validation/test samples and out-of-domain abstention are explicit.
The small tanh random-feature network learns its output weights by ridge
regression. It never replaces collision or analytical semantic acceptance.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np

from .scene_spec import PROJECT_ROOT, load_scene_config
from .whole_robot_room import WholeRobotRoomOracle


def signature(scene, region, margin):
    data = {key: scene.config[key] for key in ('robot', 'arm', 'gripper')}
    data.update(region=scene.config['semantic_regions'][region], margin=margin,
                drake=importlib.metadata.version('drake'))
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def predict(model, qs):
    qs = np.atleast_2d(qs)[:, :10]
    x = (qs-np.asarray(model['mean']))/model['scale']
    hidden = np.tanh(x @ np.asarray(model['hidden_weights']) + model['hidden_bias'])
    design = np.c_[np.ones(len(x)), x, hidden]
    return design @ np.asarray(model['output_weights'])


def train(scene, output, *, region='room2', margin=.02, sizes=(2400, 600, 600), seed=20261020):
    if len(sizes) != 3 or any(type(n) is not int or n < 20 for n in sizes):
        raise ValueError('Three split sizes >=20 required')
    bounds = scene.config['semantic_regions'][region]
    oracle = WholeRobotRoomOracle.from_xy_bounds(scene, bounds['x_bounds'], bounds['y_bounds'], margin=margin)
    tucked = np.asarray(scene.config['pick_task']['navigation_arm_q'])
    lower = np.r_[[-.2, -.8, -.4], np.maximum(tucked-.65, scene.config['whole_body_planning']['joint_lower_limits'])]
    upper = np.r_[[1.3, .8, .4], np.minimum(tucked+.65, scene.config['whole_body_planning']['joint_upper_limits'])]
    datasets = []
    for split, n in enumerate(sizes):
        rng = np.random.default_rng(seed+split)
        qs = rng.uniform(lower, upper, (n, 10))
        ys = np.array([oracle.query(q)['residual_m'] for q in qs])
        datasets.append((qs, ys))
    x, y = datasets[0]
    rng = np.random.default_rng(seed+10)
    model = {'version': 1, 'architecture': '10-input / 192-tanh-random-features / trained-linear-output',
        'mean': x.mean(axis=0).tolist(), 'scale': x.std(axis=0).tolist(),
        'hidden_weights': rng.normal(0, .35, (10, 192)).tolist(),
        'hidden_bias': rng.normal(0, .5, 192).tolist(),
        'lower': lower.tolist(), 'upper': upper.tolist(), 'region': region, 'margin_m': margin,
        'scene_signature': signature(scene, region, margin), 'seed': seed,
        'split_seeds': [seed+i for i in range(3)], 'split_sizes': list(sizes),
        'scope': 'Geometric room residual surrogate; not learned human preference; analytical guard mandatory'}
    standardized = (x-model['mean'])/model['scale']
    design = np.c_[np.ones(len(x)), standardized,
                   np.tanh(standardized@np.asarray(model['hidden_weights'])+model['hidden_bias'])]
    weights = np.linalg.solve(design.T@design + .01*np.eye(design.shape[1]), design.T@y)
    model['output_weights'] = weights.tolist()
    validation_prediction = predict(model, datasets[1][0])
    model['abstention_margin_m'] = float(np.quantile(np.abs(validation_prediction-datasets[1][1]), .95))
    metrics = {}
    for name, (qs, ys) in zip(('train', 'validation', 'test'), datasets):
        predicted = predict(model, qs)
        confident = np.abs(predicted) > model['abstention_margin_m']
        metrics[name] = {'samples': len(qs), 'rmse_m': float(np.sqrt(np.mean((predicted-ys)**2))),
            'sign_accuracy': float(np.mean((predicted >= 0) == (ys >= 0))),
            'positive_labels': int(np.sum(ys >= 0)), 'confident_fraction': float(np.mean(confident)),
            'confident_false_acceptances': int(np.sum(confident & (predicted >= 0) & (ys < 0))),
            'dataset_sha256': hashlib.sha256(np.c_[qs, ys].tobytes()).hexdigest()}
    model['metrics'] = metrics
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2))
    np.savez_compressed(output.with_suffix('.data.npz'), **{
        f'{name}_{key}': array for name, pair in zip(('train', 'validation', 'test'), datasets)
        for key, array in zip(('q', 'residual'), pair)})
    return model


class GuardedLearnedRoom:
    def __init__(self, scene, spec):
        path = PROJECT_ROOT/Path(spec['model_path'])
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != spec['model_sha256']:
            raise ValueError('Learned semantic model hash mismatch')
        self.model = json.loads(payload)
        self.scene = scene
        if (self.model.get('version') != 1 or self.model['region'] != spec['region']
                or self.model['margin_m'] != spec.get('margin_m', 0.)
                or self.model['scene_signature'] != signature(scene, spec['region'], spec.get('margin_m', 0.))):
            raise ValueError('Learned semantic model scene/domain mismatch')
        for key, shape in [('mean', (10,)), ('scale', (10,)), ('lower', (10,)), ('upper', (10,)),
                           ('hidden_weights', (10, 192)), ('hidden_bias', (192,)), ('output_weights', (203,))]:
            array = np.asarray(self.model[key])
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError('Invalid learned semantic model weights/domain')
        if (np.any(np.array(self.model['scale']) <= 0) or np.any(np.array(self.model['lower']) >= self.model['upper'])
                or not np.isfinite(self.model['abstention_margin_m']) or self.model['abstention_margin_m'] < 0):
            raise ValueError('Invalid learned semantic normalization/domain')
        bounds = scene.config['semantic_regions'][spec['region']]
        self.oracle = WholeRobotRoomOracle.from_xy_bounds(scene, bounds['x_bounds'], bounds['y_bounds'],
            margin=spec.get('margin_m', 0.))

    def evaluate(self, q):
        in_domain = bool(np.all(q[:10] >= self.model['lower']) and np.all(q[:10] <= self.model['upper']))
        predicted = float(predict(self.model, q)[0]) if in_domain else None
        abstained = not in_domain or abs(predicted) <= self.model['abstention_margin_m']
        exact = self.oracle.query(q[:10], q[10:])
        return {'satisfied': bool(exact['satisfied']), 'residual': exact['residual_m'],
            'method': 'learned_surrogate_with_analytical_guard', 'prediction_m': predicted,
            'abstained': bool(abstained), 'out_of_domain': not in_domain,
            'surrogate_disagreed': predicted is not None and (predicted >= 0) != exact['satisfied'],
            'scope': 'Surrogate diagnostic/ranking; acceptance always checked by exact shape support'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/scene_semantic_choice_v6.yaml')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from .build_scene import build_scene
    model = train(build_scene(config=load_scene_config(args.config), visualize=False), args.output)
    print(json.dumps(model['metrics'], indent=2))
