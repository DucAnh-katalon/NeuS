from exp_runner import Runner
import torch
from tqdm import tqdm
import trimesh
import os
import numpy as np
import logging, argparse

class CustomRunner(Runner):
    def __init__(self,  conf_path, mode='train', case='CASE_NAME', is_continue=False):
        super().__init__( conf_path, mode, case, is_continue)

    def validate_mesh_vertex_color(self, world_space=False, resolution=64, threshold=0.0, name=None):
        print('Start exporting textured mesh')

        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        vertices, triangles = self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution,
                                                            threshold=threshold)
        print(f'Vertices count: {vertices.shape[0]}')

        vertices = torch.tensor(vertices, dtype=torch.float32)
        vertices_batch = vertices.split(self.batch_size)
        render_iter = len(vertices_batch)

        vertex_colors = []
        for iter in tqdm(range(render_iter)):
            feature_vector = self.sdf_network.sdf_hidden_appearance(vertices_batch[iter])[:, 1:]
            gradients = self.sdf_network.gradient(vertices_batch[iter]).squeeze()
            dirs = -gradients
            vertex_color = self.color_network(vertices_batch[iter], gradients, dirs,
                                                feature_vector).detach().cpu().numpy()[..., ::-1]  # BGR to RGB
            vertex_colors.append(vertex_color)
        vertex_colors = np.concatenate(vertex_colors)
        print(f'validate point count: {vertex_colors.shape[0]}')
        vertices = vertices.detach().cpu().numpy()

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)
        if name is not None:
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', f'{name}.ply'))
        else:            
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_vertex_color.ply'.format(self.iter_step)))

        logging.info('End')


if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    # runner = Runner(args.conf, args.mode, args.case, args.is_continue)
    runner = CustomRunner(args.conf, args.mode, args.case, args.is_continue)
    runner.validate_mesh_vertex_color(threshold=args.mcube_threshold, resolution=512)