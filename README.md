# Game of Cuda - A Song of Blocks and Threads
_*Click on image to watch the demo on YouTube.*_
<br>(P.S. Select 4K quality for best viewing experience)
<br>[![4K video demo of Game of Cuda - A Song of Blocks and Threads](https://img.youtube.com/vi/NsOmfuF-NtU/0.jpg)](https://www.youtube.com/watch?v=NsOmfuF-NtU)

## Overview
Realtime, 60 FPS, implementation of Conway's Game of Life using CUDA.
It's not the most efficient implementation and there is still lots of room for improvement.

## Controls
- **Left Mouse Movement + Button (Hold)**: Pan
- **Mouse Scroll**: Zoom
- **Arrow Keys**: Increase/Decrease Blur Passes
- **W/S**: Increase/Decrease Exposure
- **A/D**: Increase/Decrease Gamma
- **Left/Right**: Increase/Decrease Threshold

##  Memory & Performance Enhancement Ideas
- [ ] Bloom kernels are applied to entire texture, instead apply only in screen space.
- [ ] Better usage of texture memory, reads, and writes.
- [ ] Grid and block size optimization.
- [ ] Kernel optimization.
  - [ ] Could the game of life kernel benefit from dynamic parallelism?
- [ ] Is the CUDA OpenGL interop as efficient as it could be?
- [ ] Bit packing to represent multiple cells in a single byte.
- [ ] Is there a way to avoid the need to double buffer the simulation in a highly parallelized execution?

## References
- [CUDA OpenGL Interop](https://medium.com/@fatlip/cuda-opengl-interop-e4edd8727c63)
  - Great flow chart explaining communication between CUDA and OpenGL.