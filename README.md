# Destruction-Cuda-OpenGL
3D sphere cloud simulation software for modeling particle based systems.

This software will allow the user to model massive dynamic particle based systems with fully programmable behavior.

Completed Tasks:
 - Implemented a GPU based Parallel Linear Bounding Volume Hierarchy to accelerate simulation calculations and ray tracing capabilities using a rapid parallel onstruction algorithm in CUDA.
 - Implemented sphere shaders which are able to rapidly draw millions of spheres from point primitives in a single draw call.
 - Implemented simple kinematics that can simulate motion for 16 million objects each frame.

Completed but unused features:
 - Implemented a ray tracer with the capability to rapidly calculate basic shadows and reflections and display the simulation model using CUDA, and OpenGL.

Current Task:
 - Use the parallel bounding volume hierarchy to simulate collisions

Next Tasks:
 - Use bounding volume hierarchy to simulate other types of physical systems.
