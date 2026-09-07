"""The Isaac Sim / Isaac Lab side of the simulator.

Everything in here has to be imported *after* a ``SimulationApp`` exists, which
is why it is a separate package rather than part of ``lidar_demo.sim``: importing
``omni`` or ``pxr`` before Kit has started fails, and the plain numpy backend must
stay importable on a machine with no Isaac at all.

``lidar_demo.sim.isaac.record`` is the entry point that starts the app first and
then does everything else.
"""
