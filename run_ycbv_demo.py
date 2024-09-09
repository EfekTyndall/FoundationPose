# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--ycbv_dir', type=str, default="/home/martyn/Thesis/ycbv", help="data dir")
  parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
  # Add argument for a video specific video sequence
  parser.add_argument('--video_id', type=str, default=None, help='ID of the video sequence to process')
  # Add argument for a specific object
  parser.add_argument('--object_id', type=int, default=None, help='ID of the object to process')
  parser.add_argument('--ref_view_dir', type=str, default="/home/martyn/Thesis/ycbv-ref/ref_views_16")
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()
  os.environ["YCB_VIDEO_DIR"] = args.ycbv_dir

  set_logging_format()
  set_seed(0)

  # Path to the specific video sequence
  video_dir = os.path.join(args.ycbv_dir, 'test', args.video_id)

  # Initialize reader for YCB-Video
  reader = YcbVideoReader(video_dir, zfar=1.5)

  # Load the mesh for the specified object (either reconstructed or ground-truth)
  if args.use_reconstructed_mesh:
      mesh = reader.get_reconstructed_mesh(args.object_id, ref_view_dir=args.ref_view_dir)
  else:
      mesh = reader.get_gt_mesh(args.object_id)
  
  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis_ycb_demo {debug_dir}/ob_in_cam_ycb_demo')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)

    # Check if the specified object is present in the current frame
    scene_ob_ids = reader.get_instance_ids_in_image(i)
    if args.object_id not in scene_ob_ids:
      print(f"Object {args.object_id} not found in frame {i}. Skipping...")
      continue

    if i==0:
      mask = reader.get_mask(i, ob_id=args.object_id, type='mask').astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    os.makedirs(f'{debug_dir}/ob_in_cam_ycb_demo', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam_ycb_demo/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis_ycb_demo', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis_ycb_demo/{reader.id_strs[i]}.png', vis)

