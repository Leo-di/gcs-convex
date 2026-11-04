"""ReplicaCAD geometry conversion and fixed-environment Drake integration.

Furniture retains the publisher's separate convex collision meshes. Articulated
assets are imported as rigid snapshots at their declared/default joint values.
The architecture is decomposed separately; no whole-apartment hull is used.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

REVISION = '3e8c7fe5759f64bfcbc3882f9cdf6de97f82a06d'
Y_UP_TO_Z_UP = np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])


def write_visual(scene,path):
    import trimesh
    data=trimesh.exchange.gltf.export_gltf(scene,embed_buffers=True,merge_buffers=True)
    path.write_bytes(data['model.gltf'])


def transform(translation=(0,0,0), quaternion=(1,0,0,0)):
    X = np.eye(4)
    X[:3,:3] = Rotation.from_quat(np.asarray(quaternion)[[1,2,3,0]]).as_matrix()
    X[:3,3] = translation
    return X


def glb_parts(path):
    import trimesh
    scene = trimesh.load(path, force='scene')
    for node in scene.graph.nodes_geometry:
        matrix, key = scene.graph[node]
        mesh = scene.geometry[key]
        yield trimesh.transform_points(mesh.vertices, matrix), mesh.faces.copy()


def convex_file(vertices, path):
    import trimesh
    vertices = np.asarray(vertices)
    if len(vertices) < 4 or np.linalg.matrix_rank(vertices-vertices.mean(0), tol=1e-9) < 3:
        return None
    hull = trimesh.Trimesh(vertices=vertices).convex_hull
    hull.export(path)
    return str(path.resolve())


def prepare_furniture(root, template, output, *, size=None):
    """Return mesh paths in a common body frame, optionally normalized to size."""
    import trimesh
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root/f'configs/objects/{template}.object_config.json').read_text())
    visual = (root/'configs/objects'/cfg['render_asset']).resolve()
    B = np.eye(4)
    B[:3,:3] = Y_UP_TO_Z_UP
    B[:3,3] = -Y_UP_TO_Z_UP @ np.asarray(cfg.get('COM', [0,0,0]))
    scene = trimesh.load(visual, force='scene')
    scene.apply_transform(B)
    if size is not None:
        bounds = scene.bounds
        if cfg.get('collision_asset'):
            collision = (root/'configs/objects'/cfg['collision_asset']).resolve()
            points = np.vstack([trimesh.transform_points(v,B) for v,_ in glb_parts(collision)])
            bounds = np.array([points.min(0),points.max(0)])
        scale = np.asarray(size)/(bounds[1]-bounds[0])
        N = np.eye(4); N[:3,:3] = np.diag(scale); N[:3,3] = -scale*bounds.mean(0)
        scene.apply_transform(N); B = N @ B
    visual_out = output/'visual.gltf'; write_visual(scene,visual_out)
    pieces = []
    if cfg.get('use_bounding_box_for_collision'):
        X=np.eye(4); X[:3,3]=scene.bounds.mean(0)
        pieces.append(dict(type='box',size=np.ptp(scene.bounds,axis=0).ravel().tolist(),pose=X.tolist()))
    else:
        collision = (root/'configs/objects'/cfg['collision_asset']).resolve()
        for i, (vertices, _) in enumerate(glb_parts(collision)):
            v = trimesh.transform_points(vertices, B)
            path = convex_file(v, output/f'convex_{i:03d}.obj')
            if path:
                pieces.append(dict(type='convex', path=path))
    result = dict(template=template, visual=str(visual_out.resolve()), collisions=pieces,
        source=cfg, bounds=scene.bounds.tolist(), normalized_size=size,
        source_revision=REVISION, geometry_method='Publisher convex components retained separately')
    (output/'asset.json').write_text(json.dumps(result, indent=2))
    return result


def components(vertices, faces):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    v, inv = np.unique(np.round(vertices, 7), axis=0, return_inverse=True)
    f = inv[faces]
    edges = np.vstack([f[:,[0,1]], f[:,[1,2]], f[:,[2,0]]])
    _, labels = connected_components(coo_matrix((np.ones(len(edges)), (edges[:,0],edges[:,1])),
        shape=(len(v),len(v))), directed=False)
    for label in np.unique(labels[f[:,0]]):
        ff = f[labels[f[:,0]] == label]
        ids, local = np.unique(ff, return_inverse=True)
        yield v[ids], local.reshape(-1,3)


def prepare_stage(path, output):
    """Thin triangle prisms preserve architectural openings; CoACD for detail."""
    import trimesh
    import coacd
    output.mkdir(parents=True, exist_ok=True)
    visual = trimesh.load(path, force='scene')
    visual.apply_transform(transform() @ np.block([[Y_UP_TO_Z_UP,np.zeros((3,1))],[np.zeros((1,3)),np.ones((1,1))]]))
    # Ceiling removed from the illustration only for the top-down cutaway.
    for node in list(visual.graph.nodes_geometry):
        T, key = visual.graph[node]
        v = trimesh.transform_points(visual.geometry[key].vertices, T)
        if v[:,2].min() > 2.9:
            visual.graph.transforms.remove_node(node)
    visual_out = output/'cutaway.gltf'; write_visual(visual,visual_out)
    pieces, records = [], []
    def add(v, method):
        name = output/f'part_{len(pieces):04d}.obj'
        p = convex_file(v, name)
        if p:
            pieces.append(dict(type='convex', path=p)); records.append(method)
    for vertices, faces in glb_parts(path):
        vertices = vertices @ Y_UP_TO_Z_UP.T
        for v, f in components(vertices, faces):
            # Original apartment floor is replaced by the explicit floor slab
            # at z=0.015; other architectural geometry stays collision-active.
            if v[:,2].max() < .016:
                continue
            if len(f) > 1000:
                print('Decomposing architectural detail:',len(f),'triangles',flush=True)
                coacd.set_log_level('error')
                parts = coacd.run_coacd(coacd.Mesh(v,f), threshold=.025, real_metric=True,
                    max_convex_hull=128, resolution=500, mcts_iterations=30,
                    mcts_nodes=10, preprocess_resolution=30, seed=0)
                for pv, _ in parts:
                    add(pv, 'CoACD architectural detail, 0.025 m requested threshold; numerical approximation')
            elif np.ptp(v,axis=0).max() < 2.:
                add(v, 'Conservative hull of a disconnected architectural component')
            else:
                for triangle in v[f]:
                    normal = np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
                    length = np.linalg.norm(normal)
                    if length < 1e-10:
                        continue
                    normal *= .001/length
                    add(np.vstack([triangle-normal,triangle+normal]), 'Architectural triangle extruded 1 mm per side')
    result = dict(visual=str(visual_out.resolve()), collisions=pieces,
        methods=records, floor_replacement_top_m=.015, ceiling_visual='cutaway only; collision retained')
    (output/'asset.json').write_text(json.dumps(result,indent=2))
    return result


def urdf_snapshot(root, instance, output):
    """Bake URDF link geometry with forward kinematics at initial joint values."""
    import trimesh
    name = instance['template_name']
    candidates = list((root/'urdf').glob(f'*/{name}.urdf'))
    if len(candidates) != 1:
        raise ValueError(f'Ambiguous articulated asset: {name}')
    source = candidates[0]; robot = ET.parse(source).getroot()
    joints = robot.findall('joint'); child_links = {j.find('child').get('link') for j in joints}
    base = next(l.get('name') for l in robot.findall('link') if l.get('name') not in child_links)
    poses = {base:np.eye(4)}
    def origin(element):
        X = np.eye(4)
        if element is not None:
            X[:3,3] = np.fromstring(element.get('xyz','0 0 0').replace(',',' '),sep=' ')
            X[:3,:3] = Rotation.from_euler('xyz',np.fromstring(element.get('rpy','0 0 0').replace(',',' '),sep=' ')).as_matrix()
        return X
    pending = list(joints)
    while pending:
        ready = [j for j in pending if j.find('parent').get('link') in poses]
        if not ready:
            raise ValueError('Invalid URDF tree')
        for j in ready:
            if instance.get('joint_positions'):
                raise ValueError('Non-default articulated joint initialization requires explicit mapping')
            poses[j.find('child').get('link')] = poses[j.find('parent').get('link')] @ origin(j.find('origin'))
            pending.remove(j)
    output.mkdir(parents=True,exist_ok=True)
    combined = trimesh.Scene(); collision_specs = []
    scale = float(instance.get('uniform_scale',1.))
    S = np.diag([scale,scale,scale,1.])
    for link in robot.findall('link'):
        for i,item in enumerate(link.findall('visual')):
            mesh = item.find('geometry/mesh')
            if mesh is not None:
                asset = trimesh.load(source.parent/mesh.get('filename'),force='scene')
                M=np.eye(4);M[:3,:3]=np.diag(np.fromstring(mesh.get('scale','1 1 1').replace(',',' '),sep=' '))
                asset.apply_transform(S @ poses[link.get('name')] @ origin(item.find('origin')) @ M)
                combined.add_geometry(asset, node_name=f'{link.get("name")}_{i}')
        for item in link.findall('collision'):
            X = S @ poses[link.get('name')] @ origin(item.find('origin'))
            geom = item.find('geometry')
            if geom.find('box') is not None:
                box = geom.find('box'); dims=np.fromstring(box.get('size').replace(',',' '),sep=' ')*scale
                R=X[:3,:3]/scale; X[:3,:3]=R
                collision_specs.append(dict(type='box',size=dims.tolist(),pose=X.tolist()))
            elif geom.find('mesh') is not None:
                mesh=geom.find('mesh'); M=np.eye(4); M[:3,:3]=np.diag(np.fromstring(mesh.get('scale','1 1 1').replace(',',' '),sep=' '))
                for v,_ in glb_parts(source.parent/mesh.get('filename')):
                    p=convex_file(trimesh.transform_points(v,X @ M),output/f'convex_{len(collision_specs):03d}.obj')
                    if p:collision_specs.append(dict(type='convex',path=p))
            else:
                raise ValueError(f'Unsupported collision shape in {source}')
    visual=output/'visual.gltf';write_visual(combined,visual)
    return dict(visual=str(visual.resolve()),collisions=collision_specs,
                source_urdf=str(source),mode='Rigid snapshot of URDF at default joint values')


def add_fixed_asset(plant, name, asset, pose):
    from pydrake.geometry import Mesh, Convex, Box
    from pydrake.math import RigidTransform, RollPitchYaw
    from pydrake.multibody.tree import SpatialInertia, UnitInertia, default_model_instance
    from pydrake.multibody.plant import CoulombFriction
    body=plant.AddRigidBody(name,default_model_instance(),SpatialInertia(1.,np.zeros(3),UnitInertia.SolidSphere(.1)))
    # Meshcat's glTF loader maps glTF Y-up into its mesh frame. Converted
    # vertices already share Drake's Z-up collision frame; compensate once.
    X_BV=RigidTransform(RollPitchYaw(-np.pi/2,0,0),[0,0,0])
    plant.RegisterVisualGeometry(body,X_BV,Mesh(asset['visual']),name+'_visual',[.7,.7,.7,1.])
    for i,part in enumerate(asset['collisions']):
        shape=Convex(part['path']) if part['type']=='convex' else Box(*part['size'])
        X=RigidTransform(np.asarray(part.get('pose',np.eye(4))))
        plant.RegisterCollisionGeometry(body,X,shape,f'{name}_collision_{i}',CoulombFriction(.9,.8))
    plant.WeldFrames(plant.world_frame(),body.body_frame(),RigidTransform(np.asarray(pose)))
    return body
