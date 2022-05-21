# This is a personal Notebook for Differentiable/Inverse Rendering written by ZiHao Wang. 

## Basic Knowledge for Rendering

*I will finish this part if I have time. For those who are not familiar to render, I personally strongly recommend you to watch GAMES101 in BiliBili.*

## Physical Based Differentiable Rendering

*The timeline of Differentiable Rendering is according to the papers I have read and I will put the related papers somewhere else.*

We start by only considering primary visibility where all obejects in scenes are projected into the camera screen space (2D) utilizing camera pose. Then color of a pixel $I$ is integral over pixel filter $k$ and radiance $L$. According to render function, the color $I$ can be written as : 
$$I = \iint f(x,y;\Phi)dxdy$$ 
where we combine pixel filter and radiance into scene function $f(x,y)=k(x,y)L(x,y)$ and $\Phi$ are some rendering parameters we are interested in, such as position of a mesh vertex. So the gradients of the intergal with respect to $\Phi$ can be written as:
$$\nabla I = \nabla \iint f(x,y;\Phi)dxdy$$ 

Notice that the integral usually dosen't contain a closed-form solution so we rely on Monte Carlo integration to estimate the pixel color $I$ which is commonly used in Physical Based Rendering.(P.S. SG is also used to approximate the rendering function but in a closed-form solution, which will be introduced later.) However we cannot take the naive approach of applying the same Monte Carlo sampler to estimate the gradient $\nabla I$, since the scene function $f$ is not necessarily differentiable with respect to scene parameters. A direct example is shown below.

<p align="center"> 
<img src="https://github.com/AndyWangZH/Differentiable-Rendering-Notebook/blob/main/image/Redner_sampling.png">
</p>

We are interested in computing the derivative of the color of pixel $I$ with respect to the green triangle vertices $v$ moving up. When using traditional area sampling (yellow samples), $I(v)$ is a step function of $v$ which is not continous and not differentiable at the point of discontinuity $v^*$. **A key observation is that all discontinuities happen at triangle edges.** So Tzu-Mao Li explicitly integrate over the discontinuities and here I give a detailed description about their works. A 2D triangle edge (as we consider primary visibility in screen space) splits the space into two half-spaces ($f_u$ and $f_l$) and we can model it as a Heaviside step function $\theta$:
$$\theta(\alpha(x,y))f_u(x,y) + \theta(-\alpha(x,y))f_l(x,y))$$ 

<p align="center"> 
<img src="https://github.com/AndyWangZH/Differentiable-Rendering-Notebook/blob/main/image/Redner_halfspace.png#pic_center">
</p>

where $f_u$ represents the upper half-space, $f_l$ represents the lower
half-space and $\alpha(x,y)$ defines the edge equation formed by the triangles. For each edge with two endpoints $(a_x, a_y), (b_x,b_y)$, we can construct the edge equation by forming the line: 
$$\alpha(x,y) = (a_y - b_y)x + (b_x - a_x)y + (a_xb_y - b_xa_y)$$

We can rewrite the scene function $f$ as a summation of Heaviside step functions $\theta$ with edge equation $\alpha_i$ multiplied by an arbitrary function $f_i$:
$$\iint f(x,y)dxdy = \sum_{i} \iint \theta(\alpha_i(x,y))f_i(x,y)dxdy $$

We want to analytically differentiate the Heaviside step function $\theta$ and explicitly integrate over its derivative – the Dirac delta function   $\delta$. To do this we first swap the gradient operator inside the integral, then we use product rule to separate the integral into two:
$$\nabla \iint \theta(\alpha_i(x,y))f_i(x,y)dxdy$$
$$= \iint \delta(\alpha_i(x,y))\nabla \alpha_i(x,y) f_i(x,y)dxdy$$
$$+ \iint \nabla f_i(x,y) \theta(\alpha_i(x,y))dxdy$$

We can estimate the gradient using two Monte Carlo estimators. The first one estimates the integral over the edges of triangles containing the Dirac delta functions, and the second estimates the original pixel integral except the smooth function $f_i$ is replaced by its gradient, which can be computed through automatic differentiation.

To estimate the integral containing Dirac delta functions, we eliminate the Dirac function by performing variable substitution to rewrite the first term containing the Dirac delta function to an integral that integrates over the edge, that is, over the regions where $\alpha_i(x,y) = 0$:
$$\iint \delta(\alpha_i(x,y))\nabla \alpha_i(x,y) f_i(x,y)dxdy$$
$$= \int_{\alpha_i(x,y)=0} \frac{\nabla \alpha_i(x,y)}{\lVert \nabla_{x,y} \alpha_i(x,y) \rVert} f_i(x,y)d\sigma(x,y)$$

where $\lVert \nabla_{x,y} \alpha_i(x,y) \rVert$ is the $L^2$ length of the gradient of the edge equations $\alpha_i$ with respect to $x,y$, which takes the Jacobian of the variable substitution into account. $\sigma(x,y)$ is the measure of the length on the edge. (*Sorry for my poor math as I currently can't understand how this variable substitution is done and how can this help for differentiable.*)

The gradients of the edge equations $\alpha_i$ are:
$$\lVert \nabla_{x,y} \alpha_i \rVert = \sqrt{(a_x - b_x)^2 + (a_y - b_y)^2}$$
$$\frac{\partial a_i}{\partial a_x} = b_y - y, \frac{\partial a_i}{\partial a_y} = x - b_x$$
$$\frac{\partial a_i}{\partial b_x} = y - a_y, \frac{\partial a_i}{\partial b_y} = a_x - x$$
$$\frac{\partial a_i}{\partial x} = a_y - b_y, \frac{\partial a_i}{\partial y} = b_x - a_x$$

As a byproduct of the derivation, we also obtain the screen space gradients $\partial x$ and $\partial y$. We can obtain the gradient with respect to other parameters, such as camera parameters, 3D vertex positions, or vertex normals by propagating the derivatives from the projected triangle vertices
using the chain rule:
$$\frac{\partial \alpha}{\partial p} = \sum_{k \in (x,y)} \frac{\partial \alpha}{\partial a_k} \frac{\partial a_k}{\partial p} + \frac{\partial \alpha}{\partial b_k} \frac{\partial b_k}{\partial p}$$
where $p$ is the desired parameter.

We use Monte Carlo sampling to estimate the Dirac integral. Recall that a triangle edge defines two half-spaces, therefore we need to compute the two values $f_l(x,y)$ and $f_u(x,y)$ on the edge. Monte Carlo estimation of the Dirac integral for a single edge $E$ on a triangle can be written as:
$$\frac{1}{N}\sum_{j=1}^N \frac{\lVert E \rVert\nabla \alpha_i(x_j,y_j)(f_u(x_j,y_j) - f_l(x_j,y_j))}{P(E) \lVert \nabla_{x_j,y_j} \alpha_i(x_j,y_j) \rVert}$$

where $\lVert E \rVert$ is the length of the edge and $P(E)$ is the probability of selecting edge $E$.

In practice, if we employ smooth shading, most of the triangle edges are in the continuous regions and the Dirac integral is zero for these edges since by definition of continuity $f_u (x,y) = f_l (x,y)$. Only the silhouette edges have non-zero contribution to the gradients. We select the edges by projecting all triangle meshes to screen space and clip them against the camera frustrum. We select one silhouette edge with probability proportional to the screen space lengths. We then uniformly pick a point on the selected edge. Our method handles occlusion correctly, since if the sample is blocked by another surface, $(x,y)$ will always land on the continuous part of the contribution function $f(x,y)$. Such samples will have zero contribution to the gradients. To recap, we use two sampling strategies to estimate the gradient integral of pixel filter: one for the discontinuous regions of the integrand, one for the continuous regions. To compute the gradient for discontinuous regions, we need to explicitly sample the edges. We compute the difference between two sides of the edges using Monte
Carlo sampling.

(*For secondary visibility, coming soon.*)