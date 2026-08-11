# Third-party notices

This extension is an independent migration and contains adaptations informed by:

- WMP (`WMP/`): Apache-2.0 overall; its vendored legged-gym portions are
  BSD-3-Clause, DreamerV3-Torch portions are MIT, and Parkour terrain portions
  are MIT. The upstream license files remain in `WMP/licenses/`.
- IsaacLab APIs and task patterns: BSD-3-Clause. IsaacLab itself is not copied
  into this extension and remains an external dependency.
- The packaged Unitree Go2 USD description: Unitree Robotics BSD-3-Clause. The
  full notice is stored beside the asset at
  `source/wmp_go2/wmp_go2/assets/robots/go2/LICENSE.txt`.
- `legged_mpc_amp`: BSD-3-Clause, used as the Go2 description/CSV contract and
  future independent data-collection fallback. Its notice is stored beside the
  asset and in the unchanged upstream reference repository.

No ROS, Gazebo, NMPC, or WBC source is copied into the Python extension.
