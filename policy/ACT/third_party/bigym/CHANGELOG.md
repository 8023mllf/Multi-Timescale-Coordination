# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- None.

### Changed

- None.

### Fixed

- None.

## 4.2.0

### Added

- Simplified the process of adding new models with the introduction of robot configurations.
- Enabled the ability to override the robot in the environment.

## 4.1.0

### Added

- Added DemoPlayer entity containing shortcuts to replay and validate demonstrations.

### Changed

- Adjusted stiffness of floating base positional actuators to reach target state with +-1mm precision.

## 4.0.0

### Added

- Added configurable floating base DOFs. By default, floating base has [X, Y, RZ] DOFs.
- Added new tasks.
- Added presets for environments.

### Changed

- Different decimation for absolute and delta action modes.

## 3.0.0

### Changed

- BiGym refactored to use [Mojo](https://github.com/stepjam/mojo).

## 2.12.0

### Added
- Added customizable `control_frequency` of the `BiGymEnv`.

### Changed
- Updated Demo Player.

## 2.11.0

### Changed

- Environment reset behaviour should be implemented in `on_reset` now instead of `reset_model`.
- Environment termination behaviour uses `_should_terminate` which should now be overridden in the task environment.

## 2.10.0

### Added

- Lightweight demos are automatically uploaded to Google Cloud Buckets when using `DemoStore.upload_demo` and `DemoStore.upload_demos`.
- `TooManyDemosRequestedError` is raised when requesting more demos than available in the bucket.
- `DemoNotFoundError` is raised when requesting demos when none are available in the bucket.

## 2.9.0

### Changed

- The observation vector of the `BiGymEnv` is now configured
  using `ObservationConfig` and `CameraConfig` instead of flags.

## 2.8.0

### Added

- Added `DemoStore` to upload and download demonstrations from Google Cloud Buckets.
- Added `Lightweight` demo format to store only demo action, termination and truncation information.

## 2.7.1

### Added

- Implemented handling of `PhysicsError` resulting from an unstable state in the simulation.
  Instead of raising an error, `UnstableSimulationWarning` is now raised.
  If multiple `UnstableSimulationWarning` occur consecutively, `UnstableSimulationError` is raised.
  The `is_healthy` property of the `BiGymEnv` now indicates the current state of the simulation.

### Changed

- Refactored utils file into separate modules.

### Fixed

- Fixed NumPy 1.25 `DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated.` for `move_plate_between_drainers.py`.

## 2.6.1

### Added

- Added demo replay utility: [replay_demos.py](tools/demo_player/replay_demos.py).

### Changed

- Wrist joints are now part of the arm instead of gripper.
- Order of arm joints and actuators is matching now.
- Removed EE pose action mode.

### Fixed

- Visual observations can now be collected simultaneously with rendering to the on-screen window.

## 2.5.1

### Fixed

- Fixed `block_until_reached` flag for absolute joint position control mode.

## 2.5.0

### Added

- Additional wrist DOF could be added to H1 model.

### Changed

- Updated the control range of grippers to a discrete mode ranging from 0 to 1.

## 2.4.0

### Added

- Added EE pose action mode.

### Changed

- Renamed and updated `put_plate_in_drainer` task to `move_plate_betweeen_drainers`.
- Renamed and updated `swipe_table` task to `stack_blocks`.
- Gripper control space is now always in the 0-1 range.

### Fixed

- When floating mode is enabled redundant joints are removed completely to prevent instability.
- Removed unnecessary rotation offsets of props to simplify manipulation.

## 2.3.0

### Added

- Added functionality to run BiGym environments in VR using pyopenxr. See [collect_demos.py](demonstrations/collect_demos.py).

### Changed

- `BiGymEnv._add_dynamic_objects()` is now an abstract method and should not be called in child classes.

## 2.2.0

### Added

- Added functionality to control seed of the BiGym environments.
- Added demo collection/replaying functionality and examples.

## 2.1.0

### Added

- Added control of H1 via mediapipe body tracking.
- Added new `SwipeTable` task.

## 2.0.0

### Added

- Added the `discrete_gripper` flag to control grippers in binary open-closed mode instead of positional mode.

### Changed

- Constructor now takes an `ActionMode`. Constructor arg `floating` has been removed and rolled into new `ActionMode` workflow.
- We now make use of `mjcf.physics` rather than directly using `mujoco.MjData`.

### Fixed
- Floating base is now controlled by positional actuators, and collisions are not ignored.

## 1.2.0

### Added

- Added new examples for tracking body with kinect to control H1.

## 1.1.1

### Fixed

- Fixed "floppy" legs when H1 is controlled in floating mode.
Enabling the floating mode now "freezes" the legs of the H1; collisions are disabled, and the motion range is limited to a minimum.

## 1.1.0

### Added

- Added new "Put plate in drainer" task.
- Added `is_gripper_holding_object` to `BiGymEnv` for checking if an object is grasped.
- Added `get_ee_pos` and `get_gripper_ee_pos` to `BiGymEnv` to retrieve the positions of H1 hands and grippers.

## 1.0.2

### Added

- Attached 2 2F-85 Robotiq grippers to H1.


### Fixed

- Automatically adjust home key frame of the H1 when attaching grippers.

## 1.0.1

### Fixed

- XML files not being included when installing.

## 1.0.0

### Added

- Head, left and right wrist cameras.

### Changed

- Constructor arguments to take in camera names.

## 0.2.0

### Added

- Floating base so that H1 can be controlled without legs;  (x, y, theta) actions
- End-effector "sites".
- Added reach target task.

## 0.1.0

### Added

- Initial project code.
