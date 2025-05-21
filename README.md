# Geometry Operator Learning with Optimal Transport
In this work, we propose a novel method that integrates geometry learning into solution operator learning for partial differential equations (PDEs) on complex geometric domains. Classical geometry learning problems in machine learning (ML) typically relies on discretized meshes or point clouds to represent geometric domains. Our approach generalizes discretized meshes to mesh density functions and formulates geometry embedding as an optimal transport (OT) problem, mapping these mesh density functions to a uniform density function in a reference space.
Compared to previous methods with interpolation or shared deformation, our OT-based method has instance-dependent deformation, which is flexible and efficient. 
Further, for 3D simulations focused on surfaces, our OT-based neural operator embeds the surface geometry into 2D parameterized latent geometry. By constraining computations to the 2D representation of the surface manifold, we achieve significant computational efficiency gains compared to volumetric simulation.
Experiments utilizing Reynolds-averaged Navier–Stokes equations (RANS) on the ShapeNet-Car and DrivAerNet-Car datasets reveal that our method not only achieves superior accuracy but also significantly reduces computational expenses in terms of both time and memory usage compared to earlier machine learning models. Additionally, our model demonstrates significantly improved accuracy on the FlowBench dataset, underscoring the benefits of utilizing instance-dependent deformation for datasets with highly variable geometries.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Xinyi-Li-4869/OTNO.git
cd OTNO
```

### 2. Set Up a Virtual Environment
```
# Create environment from YAML file (recommended)
conda env create -f env.yml
conda activate otno

# OR manually create environment
conda create -n otno python=3.10
conda activate otno
conda install -f requirements.txt
```

## Run the Project
```
python shapenet/otno/ot_train.py
python drivaernet/otno/ot_train.py
python flowbench/otno/ot_train_boundary.py --resolution 512 --expand_factor 3 --group_name nurbs --latent_shape square
python flowbench/otno/ot_train_fullspace.py --resolution 512 --expand_factor 3 --group_name nurbs --latent_shape square
```
