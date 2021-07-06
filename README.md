### energy_formation_ros
Require ROS version kinetic or higher

Simulation package: rosplane for fixed-wing UAV and hector_quadrotor for rotor UAV, either install from the original repository or this repository.

### Usage 

Launch 12 fixed-wing UAV Gazebo simulation:

`roslaunch rosplane_sim twelve_fixedwing.launch`

Run energy formation program at the ebm folder:

`rosrun energy_formation energy_hierarchy_plane.py`

Launch hybrid Gazebo simulation (6 fixed-wing + 6 rotors):

`roslaunch hector_quadrotor_gazebo hybrid_world`

Run energy formation program at the ebm folder:

`rosrun energy_formation energy_hierarchy_hybrid.py`
