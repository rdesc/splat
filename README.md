# DuckieSplat: A Data-Driven Simulator for Duckietown

## Introduction 

Autonomous vehicles (AVs) are revolutionizing modern transportation, with Level 4 [1] self-driving vehicles becoming an increasingly regular presence on US roads. Despite this progress, ensuring the operational safety of AVs in safety-critical scenarios, such as adverse weather conditions or unexpected agent behaviours, remains a significant obstacle to the scalable adoption of self-driving technology. Simulation has emerged as a promising tool to improve coverage over the interesting, safety-critical distribution of driving scenarios, which can be used either for the training or evaluation of autonomous vehicles. 

The supported simulator for Duckietown is the Duckiematrix, which is created in Unity and relies on handcrafted assets to construct virtual environments for testing Duckiebots. This approach to simulation is similar to traditional driving simulators such as CARLA [2], which are handcrafted and thus rely on costly manual labour to create simulation environments. Consequently, these handcrafted environments are difficult to scale to novel environments and suffer from a significant sim-to-real gap. We illustrate the sim-to-real *appearance* gap in Figure 1. To address these limitations of handcrafted simulators, recent works have instead focused on \textit{data-driven} simulation, whereby collected real-world driving data can be leveraged to derive simulation environments, either by training a generative model to sample from the distribution of such data [3], or by employing novel view synthesis techniques to reconstruct 3D environments of the data [4-5]. 

| ![Figure 1](fig1.png) |
|:--:|
| Figure 1: **Data-driven simulators exhibit smaller visual sim-to-real gap than handcrafted simulators.** *Left*: Real image of Duckietown track in the lab. *Middle*: Rendered image of simulated Duckietown track in the Duckiematrix. *Right*: Rendered image of simulated Duckietown track in DuckieSplat. |

This project focuses specifically on utilizing a novel view synthesis technique known as 3D Gaussian Splatting [6] to synthesize a 3D virtual simulation environment for Duckiebots. Due to time constraints, we focus specifically on simulating realistic camera data of the physical Duckietown track in the lab. To this end, we propose **DuckieSplat**, a data-driven simulator for Duckietown based on 3D Gaussian Splatting that can simulate realistic RGB camera data of the physical Duckietown track *in real-time* (60 FPS). Figure 1 (right) shows a rendered view in the DuckieSplat simulator, which exhibits a visually smaller sim-to-real gap compared to the handcrafted Duckiematrix simulator. In the following section, we provide a concise introduction to 3D Gaussian Splatting so that readers can better understand the underlying algorithm used to generate DuckieSplat.

## Background: 3D Gaussian Splatting

This section explores a computer vision algorithm called 3D Gaussian Splatting that leverages data for *scene reconstruction*. Unlike generative models, which sample from a distribution learned from large datasets, scene reconstruction methods use a small set of 2D input observations of a *specific static scene* to recover its underlying 3D representation. These learned representations enable rendering novel views of the static scene, making them particularly useful for the sensor simulation application. In this project, the specific scene we choose to reconstruct is the physical Duckietown track in the lab. It is important to note in a typical 3D Gaussian Splatting pipline, the scene must be *static*; however, most driving scenes contain static (e.g., background) and dynamic (e.g., duckies) components. In this work, we assume that the full scene is static, and we leave separate modelling of static and dynamic components, for example as done in [4-5], for future work. Inconsistencies in the static scenes between images, for example Figure 1 (Left) contains fewer duckies than Figure 1 (Right), can cause issues for the 3D Gaussian Splatting. We therefore ensure that all collected images for Gaussian Splat training are of the same static scene. Notably, the image in Figure 1 (left) was not used to train the Gaussian Splat that rendered Figure 1 (right), as it contained fewer duckies than the rendered scene.

**Radiance Fields**: 3D Gaussian Splatting falls under a class of methods that learn a *radiance field*. A radiance field is a function $ F(\mathbf{x}, \mathbf{d}) = (\mathbf{c}, \sigma) $, where $ \mathbf{x} \in \mathbb{R}^3 $ is a point in 3D space, $ \mathbf{d} \in \mathbb{R}^3 $ is a unit vector representing the viewing direction, $ \mathbf{c} \in \mathbb{R}^3 $ is the RGB colour at that point and direction, and $ \sigma \in \mathbb{R} $ is the volume density at $\mathbf{x}$, which can be intuitively understood as the differential probability of a ray hitting a particle at location $\mathbf{x}$. A radiance field enables rendering novel views of a scene using techniques such as volume rendering or alpha blending. 3DGS uses an explicit representation of 3D geometry by modelling the radiance field with a discrete set of parameterized 3D Gaussians. This is in constrast to a related class of methods called NeRFs [7] that implicitly represent 3D geometry by parameterizing the radiance field with a neural network.

## Methods 

## Results

Duckietown project demo https://rdesc.dev/duckietown-project/index.html

## References

[1] SAE International, Sae levels of driving automation™ refined for clarity and international audience, Accessed: 2024-11-30, 2021. [Online]. Available: https://www.sae.org/blog/sae-j3016-update.

[2] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, “CARLA: An open urban driving simulator,” in Proceedings of the 1st Annual Conference on Robot Learning, 2017, pp. 1–16.

[3] K. Chitta, D. Dauner, and A. Geiger, “Sledge: Synthesizing driving environments with generative models and rule-based traffic,” in European Conference on Computer Vision (ECCV), 2024.

[4] Z. Yang, Y. Chen, J. Wang, et al., “Unisim: A neural closed-loop sensor simulator,” in CVPR, 2023.

[5] Y. Yan, H. Lin, C. Zhou, et al., “Street gaussians for modeling dynamic urban scenes,” in ECCV, 2024.

[6] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaussian splatting for real-time radiance field rendering,” ACM Transactions on Graphics, vol. 42, no. 4, Jul. 2023. [Online]. Available: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/.

[7] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” in ECCV, 2020.


