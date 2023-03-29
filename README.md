# Destruction-Cuda-OpenGL
3D sphere cloud simulation software for modeling particle based systems.

This software will allow the user to model massive dynamic particle based systems using sphere clouds and fully programmable behavior.

Completed Tasks:
 - Implemented a GPU based Parallel Linear Bounding Volume Hierarchy to accelerate simulation calculations and ray tracing capabilities using a rapid parallel onstruction algorithm in CUDA.
 - Implemented a ray tracer with the capability to rapidly calculate basic shadows and reflections and display the simulation model using CUDA, and OpenGL.
 - Used OpenGL and GLSL shaders to rasterize the simulation to the display for older hardware that doesn't support real-time ray tracing.
 - Implemented simple kinematics that can simulate motion for 16 million objects in real-time when using the rasterizer.

Current Task:
 - Use the parallel bounding volume hierarchy to simulate collisions, gravity, electromagnetic, and molecular dynamics between millions of particles so that together they have the macrophysical emergent behavior.
