# DuckieSplat: A Data-Driven Simulator for Duckietown

## Introduction 

Autonomous vehicles (AVs) are revolutionizing modern transportation, with Level 4 [1] self-driving vehicles becoming an increasingly regular presence on US roads. Despite this progress, ensuring the operational safety of AVs in safety-critical scenarios, such as adverse weather conditions or unexpected agent behaviours, remains a significant obstacle to the scalable adoption of self-driving technology. Simulation has emerged as a promising tool to improve coverage over the interesting, safety-critical distribution of driving scenarios, which can be used either for the training or evaluation of autonomous vehicles. 

The supported simulator for Duckietown is the Duckiematrix, which is created in Unity and relies on handcrafted assets to construct virtual environments for testing Duckiebots. This approach to simulation is similar to traditional driving simulators such as CARLA [2], which are handcrafted and thus rely on costly manual labour to create simulation environments. Consequently, these handcrafted environments are difficult to scale to novel environments and suffer from a significant sim-to-real gap. We illustrate the sim-to-real *appearance* gap in Figure 1. To address these limitations of handcrafted simulators, recent works have instead focused on *data-driven* simulation, whereby collected real-world driving data can be leveraged to derive simulation environments, either by training a generative model to sample from the distribution of such data [3], or by employing novel view synthesis techniques to reconstruct 3D environments of the data [4-5]. 

| ![Figure 1](fig1.png) |
|:--:|
| Figure 1: **Data-driven simulators exhibit smaller visual sim-to-real gap than handcrafted simulators.** *Left*: Real image of Duckietown track in the lab. *Middle*: Rendered image of simulated Duckietown track in the Duckiematrix. *Right*: Rendered image of simulated Duckietown track in DuckieSplat. |

This project focuses specifically on utilizing a novel view synthesis technique known as 3D Gaussian Splatting [6] to synthesize a 3D virtual simulation environment for Duckiebots. Due to time constraints, we focus specifically on simulating realistic camera data of the physical Duckietown track in the lab. To this end, we propose **DuckieSplat**, a data-driven simulator for Duckietown based on 3D Gaussian Splatting that can simulate realistic RGB camera data of the physical Duckietown track *in real-time* (60 FPS). Figure 1 (right) shows a rendered view in the DuckieSplat simulator, which exhibits a visually smaller sim-to-real gap compared to the handcrafted Duckiematrix simulator. In the following section, we provide a concise introduction to 3D Gaussian Splatting followed by an overview of DuckieSplat.

## DuckieSplat

This section outlines the 3D Gaussian Splatting algorithm, which leverages data for *scene reconstruction*. Unlike generative models, which sample from a distribution learned from large datasets, scene reconstruction methods use a small set of 2D input observations of a *specific static scene* to recover its underlying 3D representation. These learned representations enable rendering novel views of the static scene, making them particularly useful for the sensor simulation application. In this project, the specific scene we choose to reconstruct is the physical Duckietown track in the lab. It is important to note in a typical 3D Gaussian Splatting pipline, the scene must be *static*; however, most driving scenes contain static (e.g., background) and dynamic (e.g., duckies) components. In this work, we assume that the full scene is static, and we leave separate modelling of static and dynamic components, for example as done in [4-5], for future work. Inconsistencies in the static scenes between images, for example Figure 1 (Left) contains fewer duckies than Figure 1 (Right), can cause issues for the 3D Gaussian Splatting. We therefore ensure that all collected images for Gaussian Splat training are of the same static scene. Notably, the image in Figure 1 (left) was not used to train the Gaussian Splat that rendered Figure 1 (right), as it contained fewer duckies than the rendered scene.

**Radiance Fields**: 3D Gaussian Splatting falls under a class of methods that learn a *radiance field*. A radiance field is a function $F(\mathbf{x}, \mathbf{d}) = (\mathbf{c}, \sigma)$, where $\mathbf{x} \in \mathbb{R}^3$ is a point in 3D space, $\mathbf{d} \in \mathbb{R}^3$ is a unit vector representing the viewing direction, $\mathbf{c} \in \mathbb{R}^3$ is the RGB colour at that point and direction, and $\sigma \in \mathbb{R}$ is the volume density at $\mathbf{x}$, which can be intuitively understood as the differential probability of a ray hitting a particle at location $\mathbf{x}$. A radiance field enables rendering novel views of a scene using techniques such as volume rendering or alpha blending. 3DGS uses an explicit representation of 3D geometry by modelling the radiance field with a discrete set of parameterized 3D Gaussians. This is in constrast to a related class of methods called Neural Radiance Fields (NeRFs) [7] that implicitly represent 3D geometry by parameterizing the radiance field with a neural network.

**3D Gaussian Splatting**: 3D Gaussian Splatting (3DGS) utilizes an explicit representation of a radiance field using a discrete set of parameterized 3D Gaussians. In Figure 2, we show the Gaussian Splatting pipeline. While prior work has highlighted the flexibility of 3D Gaussians as a primitive for 3D geometry representation [8], 3DGS is the first approach to leverage these primitives for *real-time* rasterization of complex scenes. A key innovation of 3DGS is its elimination of the computational bottleneck inherent in NeRFs, which require dense neural network evaluations along rays cast from each pixel to render a single image—making real-time applications impractical. Instead, 3DGS employs a gradient-based optimization framework that entirely avoids neural network components, resulting in significantly faster training and inference compared to NeRFs. The 3D Gaussian primitive used in 3DGS is defined as $G(\mathbf{x}) = e^{-\frac{1}{2}\mathbf{x}^T\Sigma^{-1}\mathbf{x}}$, where $\mathbf{x}$ represents a 3D spatial position and $\Sigma$ is the 3D covariance matrix. 3DGS parameterizes $\Sigma$ using a scale matrix $S$ and a rotation matrix $R$:  
```math
\Sigma = RS S^T R^T,
```
which guarantees that $\Sigma$ remains positive semi-definite. The scale parameters are optimized independently along each of the $x$, $y$, and $z$ dimensions, producing an *anisotropic* covariance matrix. This anisotropic parameterization offers greater flexibility in fitting 3D geometries compared to an isotropic covariance matrix. 

| ![Figure 2](fig2.png) |
|:--:|
| Figure 2: **Gaussian Splatting Pipeline.** Gaussian Splatting starts from a sparse point cloud derived from COLMAP, which are used to initialize a set of 3D Gaussians. Given a camera view, the Gaussians are projected into 2D and rasterized efficiently with a differentiable tile rasterizer. The rendering process is fully differentiable, which enables updating the Gaussian parameters with gradient-based optimization.|

The 3D Gaussian Splatting (3DGS) pipeline begins with a Structure-from-Motion (SfM) algorithm, such as COLMAP, to estimate camera intrinsic and extrinsic parameters as well as a point cloud representing the approximate 3D geometry of the scene. This point cloud serves as the initialization for a set of 3D Gaussians, where each Gaussian is parameterized by its spatial position (*i.e.*, mean of the Gaussian), a 3D covariance matrix, an opacity value $\alpha$, and spherical harmonic coefficients that capture view-dependent color. During the rendering process, rays are cast into the scene, and the properties of 3D Gaussians intersected by each ray are retrieved. Standard $\alpha$-blending is applied along the ray to aggregate the contributions of the intersected Gaussians to derive the pixel’s RGB color. In practice, for each ray, once a target saturation is reached, the thread terminates. This efficient rendering mechanism ensures that the occluded or non-intersecting Gaussians are ignored.

The learning process in 3DGS parallels that of Neural Radiance Fields (NeRFs). Images are rendered from the ground-truth camera poses and regressed against corresponding ground-truth images to compute a reconstruction loss. The gradients of this loss are backpropagated through the rendering procedure to optimize the parameters of the 3D Gaussians. Importantly, the differentiability of the 3D-to-2D Gaussian projection and the proposed $\alpha$-blending rasterization process enables direct optimization of Gaussian parameters, without the need for any neural network components. This approach results in faster optimization and inference compared to NeRFs. The loss function utilized by 3DGS combines an L1 loss and a Structural Similarity Index Measure (SSIM) loss [9]:

```math
L = (1 - \lambda) L_1 + \lambda L_{\text{D-SSIM}}.
```

At the core of 3DGS is a *real-time* differentiable rasterization process, which allows for gradient-based optimization of the 3D Gaussian parameters. To achieve this, 3DGS bins the Gaussians into $16 \times 16$ spatial tiles, where the Gaussians in each tile are sorted by depth in parallel on a GPU using Radix sort. A separate thread is launched for each pixel to cast rays into the scene. These threads retrieve intersecting Gaussians, sorted by depth on a per-tile basis, and accumulate their contributions via $\alpha$-blending. Once a target saturation $\alpha$ is reached, the thread terminates, ensuring efficient computation. This parallelized approach drastically improves rendering speed, enabling real-time rasterization. 3DGS additionally contains an adaptive density control block that clones Gaussians with large positional gradients, and remove Gaussians with low $\alpha$ values at fixed intervals during training. 3DGS represents a significant advancement in the field of 3D computer vision, effectively replacing NeRFs as the state-of-the-art technique for novel view synthesis. Its explicit 3D representation is not only faster and more efficient but also more interpretable and easily controllable. These characteristics make 3DGS particularly well-suited for applications such as data-driven simulation, where the ability to manipulate and understand the underlying representation is critical.

**DuckieSplat**: In this project, we utilize the official 3DGS repository to train a Gaussian Splat that can reconstruct the physical Duckietown track in the lab. We call the trained Gaussian Splat **DuckieSplat**. We first require a set of images of the track that can be used to train DuckieSplat. We experiment with images collected from two types of sensors: the Duckiebot camera and an iPhone camera. For each sensor, we also experiment with two image collection protocols. The first protocol involves manually collecting images with a Duckiebot (or iPhone that is approximately positioned at the height of a Duckiebot) as it completes one loop of the track. The second protocol involves collecting images from an overhead view, where we manually hold the Duckiebot (or iPhone camera) roughly 1.5m above the track and collect images at fixed intervals while walking once around the outside of the track. Figure 3 shows an example of a collected image following the first protocol (Figure 3, Left) and second protocol (Figure 3, Right). 

Given a set of images collected using one of the two sensors and one (or both) of the image collection protocols, we first run COLMAP to attain approximate camera extrinsic and intrinsic parameters as well as a sparse point cloud of the approximate 3D geometry. We then train DuckieSplat using the official repository with default settings. Depending on the set of images used for training, COLMAP takes between 1-2 hours to run and 3DGS training takes approximately 30 minutes.

Once training is complete, we can interact with DuckieSplat in a web browser: https://rdesc.dev/duckietown-project/index.html. By default, DuckieSplat will render camera views from an (approximate) Duckiebot view as it drives along the centerline around the track in an infinite loop. Pressing the keyboard will break the infinite loop and allow a user to manually interact with DuckieSplat using keyboard control. We created the infinite loop by manually collecting a sequence of camera views (defined by a rotation matrix $R$ and translation vector $t$) while manually "driving" along the track inside DuckieSplat using keyboard control. We then interpolate the views using a simple linear interpolation scheme, where we use linear interpolation of the translation vectors and slerp interpolation of the rotations parameterized as quaternions:

```math
t_{\alpha} = (1 - \alpha) t_1 + \alpha t_2, \\
q_{\alpha} = (q_1 q_0^{-1})^{\alpha}q_0
```

In the next section, we walk through the results of DuckieSplat using the different sensors and image collection protocols, while describing the technical issues we faced in producing a high-quality reconstruction of the Duckietown track.

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

[8] M. Zwicker, H. Pfister, J. Van Baar, and M. Gross, “Ewa splatting,” IEEE Transactions on Visualization and Computer Graphics, vol. 8, no. 3, pp. 223–238, 2002.

[9] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.


