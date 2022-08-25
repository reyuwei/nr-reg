# nr-reg
Repo for non-rigid registration in **PIANO: A Parametric Hand Bone Model from Magnetic Resonance Imaging, IJCAI' 21** and **NIMBLE: A Non-rigid Hand Model with Bones and Muscles, SIG' 22**

# Usage
- Embedded non-rigid registration (PIANO)

    `reg_embedded.py`

- Tetrahedral-based non-rigid deformation (NIMBLE)

    `reg_tet.py`

## Acknowledgment
- Algorithm reference
  - [[TOG20] Constraining Dense Hand Surface Tracking with Elasticity](https://research.facebook.com/publications/constraining-dense-hand-surface-tracking-with-elasticity/)
  - [[SIG12 Course] FEM Simulation of 3D Deformable Solids: A practitioner's guide to theory, discretization and model reduction](http://viterbi-web.usc.edu/~jbarbic/femdefo/)
  - [[TVCG19] FlyFusion: Realtime Dynamic Scene Reconstruction Using a Flying Depth Camera](https://ieeexplore.ieee.org/document/8778689)
  - [[SIG18] MonoPerfCap: Human Performance Capture from Monocular Video](https://vcai.mpi-inf.mpg.de/projects/wxu/MonoPerfCap/)
  - [[SIG07] Embedded Deformation for Shape Manipulation](https://people.inf.ethz.ch/~sumnerb/research/embdef/Sumner2007EDF.pdf)


- Code reference
  - [manopth](https://github.com/hassony2/manopth/blob/master/manopth) by [Yana Hasson](https://hassony2.github.io/)
  - [pytorch_HMR](https://github.com/MandyMo/pytorch_HMR) by [Zhang Xiong](https://github.com/MandyMo)
  - [SMPLX](https://github.com/vchoutas/smplx) by [Vassilis Choutas](https://github.com/vchoutas)
  - [LoopReg](https://github.com/bharat-b7/LoopReg) by [Bharat Bhatnagar](https://github.com/bharat-b7)
  - [GAMES201-elasticity demo](https://github.com/taichi-dev/games201/releases/download/lec3/demos.zip) by [Yuanming Hu](https://yuanming.taichi.graphics/)

If you find this data useful for your research, consider citing:
```
@inproceedings{li2021piano,
        title     = {PIANO: A Parametric Hand Bone Model from Magnetic Resonance Imaging},
        author    = {Li, Yuwei and Wu, Minye and Zhang, Yuyao and Xu, Lan and Yu, Jingyi},
        booktitle = {Proceedings of the Thirtieth International Joint Conference on
                    Artificial Intelligence, {IJCAI-21}},
        editor    = {Zhi-Hua Zhou},
        pages     = {816--822},
        year      = {2021},
        month     = {8},
        note      = {Main Track},
        doi       = {10.24963/ijcai.2021/113},
        url       = {https://doi.org/10.24963/ijcai.2021/113}
}

@article{10.1145/3528223.3530079,
        author = {Li, Yuwei and Zhang, Longwen and Qiu, Zesong and Jiang, Yingwenqi and Li, Nianyi and Ma, Yuexin and Zhang, Yuyao and Xu, Lan and Yu, Jingyi},
        title = {NIMBLE: A Non-Rigid Hand Model with Bones and Muscles},
        year = {2022},
        issue_date = {July 2022},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        volume = {41},
        number = {4},
        issn = {0730-0301},
        url = {https://doi.org/10.1145/3528223.3530079},
        doi = {10.1145/3528223.3530079},
        journal = {ACM Trans. Graph.},
        month = {jul},
        articleno = {120},
        numpages = {16}
        }
```