`soma_t_pose.bvh` is a short calibration clip extracted from:

- source BVH: `/home/zxw/Documents/bones_studio_demo/soma_uniform/bvh/210531/walk_forward_professional_003__A001.bvh`
- source frame range: `5..15` inclusive
- source frame `10` corresponds to clip frame `5`

Recommended use:

```bash
.venv/bin/python tools/convert_bones_soma_bvh_to_smplx.py \
  /path/to/input.bvh \
  /path/to/output.npz \
  --calibration-bvh assets/calibration/soma_t_pose.bvh \
  --calibration-bvh-frame 5
```
