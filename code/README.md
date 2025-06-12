# Online-Non-Linear-Centroidal-MPC-with-stability-Guarantees-for-Robust-Locomotion-of-Legged-Robots-
AMR project

Simulate a locomotion on a flat surface using the MPC controller of the paper :

🔗 [Adaptive Non-linear Centroidal MPC with Stability Guarantees for Robust Locomotion of Legged Robots](https://arxiv.org/abs/2409.01144)
Mohamed Elobaid, Giulio Turrisi, Lorenzo Rapetti, Giulio Romualdi, Stefano Dafarra, Tomohiro Kawakami, Tomohiro Chaki, Takahide Yoshiike, Claudio Semini, Daniele Pucci



## 🔧 Setup & Requirements

You need Python (preferably 3.11 or 3.12) and the following dependencies:

Many packages are shared with this repo:  
🔗 [https://github.com/DIAG-Robotics-Lab/ismpc](https://github.com/DIAG-Robotics-Lab/ismpc/blob/main/README.md)


### Install using pip

```bash
pip install dartpy casadi scipy matplotlib osqp

```
it might requred numpy==1.26.0 

## Run the simulation
```
python3 project/simulation.py
```
then press spacebar to start it
