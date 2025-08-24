<h1 align="center"> Hold My Beer🍻: Learning Gentle Humanoid Locomotion and End-Effector Stabilization Control </h1>

<div align="center">

[[Website]](https://lecar-lab.github.io/SoFTA/)
<!-- [[Arxiv]](https://lecar-lab.github.io/SoFTA/) -->
<!-- [[Video]](https://www.youtube.com/) -->

<img src="assets/ip.png" style="height:50px;" />




[![IsaacGym](https://img.shields.io/badge/IsaacGym-Preview4-b.svg)](https://developer.nvidia.com/isaac-gym) [![Linux platform](https://img.shields.io/badge/Platform-linux--64-orange.svg)](https://ubuntu.com/blog/tag/22-04-lts) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()


<img src="assets/intro.gif" width="70%"/>

</div>

# TODO
- [x] Release training code 
- [x] Release evaluation code
- [ ] Release deploy code


## Installation

### IsaacGym Conda Env

Create conda environment.

```bash
conda create -n hmbgym python=3.8
conda activate hmbgym
```
#### Install IsaacGym

Download [IsaacGym](https://developer.nvidia.com/isaac-gym/download) and extract:

```bash
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4
```

Install IsaacGym Python API:

```bash
pip install -e isaacgym/python
```

Test installation:

```bash
cd isaacgym/python/examples

python 1080_balls_of_solitude.py  # or
python joint_monkey.py
```

For libpython error:

- Check conda path:
    ```bash
    conda info -e
    ```
- Set LD_LIBRARY_PATH:
    ```bash
    export LD_LIBRARY_PATH=</path/to/conda/envs/your_env/lib>:$LD_LIBRARY_PATH
    ```

#### Install SoFTA

```bash
git clone https://github.com/LeCAR-Lab/SoFTA.git
cd SoFTA

pip install -e .
pip install -e isaac_utils
```

## Training Code
### Unitree G1_27DoF
<details>
<summary>Training Command</summary>

```bash
python humanoidverse/train_agent.py +exp=async_locomotion_ma_stand_gait_ee_rrh simulator.config.sim.control_decimation=2 +opt=wandb
```

</details>

## Evaluation Code
<details>
<summary>Evaluation Command</summary>

```bash
python humanoidverse/eval_agent.py +checkpoint=<path_to_your_ckpt>
```
</details>
<details>
<summary>Interactive Commands</summary>

 - press w/a/s/d to control the linear velocity
 - press q/e to control the angular velocity
 - press z to set all commands to zero
 - press upper/lower arrow to control the EE z pos
 - press left/right arrow to control the EE y pos
 - press page up/down to control the EE x pos
 - press page 1/2 to control the gait period

</details>

## Citation
If you find our work useful, please consider citing us!

```bibtex
@article{li2025softa,
          title={Hold My Beer: Learning Gentle Humanoid Locomotion and End-Effector Stabilization Control},
          author={Li, Yitang and Zhang, Yuanhang and Xiao, Wenli and Pan, Chaoyi and Weng, Haoyang and He, Guanqi and He, Tairan and Shi, Guanya},
          journal={arXiv preprint arXiv:2505.24198},
          year={2025}
        }
```

Also consider citing these prior works that are used in this project:

```bibtex
@article{zhang2025falcon,
          title={FALCON: Learning Force-Adaptive Humanoid Loco-Manipulation},
          author={Zhang, Yuanhang and Yuan, Yifu and Gurunath, Prajwal and He, Tairan and Omidshafiei, Shayegan and Agha-mohammadi, Ali-akbar and Vazquez-Chanlatte, Marcell and Pedersen, Liam and Shi, Guanya},
          journal={arXiv preprint arXiv:2505.06776},
          year={2025}
        }
@article{he2025asap,
          title={ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills},
          author={He, Tairan and Gao, Jiawei and Xiao, Wenli and Zhang, Yuanhang and Wang, Zi and Wang, Jiashun and Luo, Zhengyi and He, Guanqi and Sobanbabu, Nikhil and Pan, Chaoyi and Yi, Zeji and Qu, Guannan and Kitani, Kris and Hodgins, Jessica and Fan, Linxi "Jim" and Zhu, Yuke and Liu, Changliu and Shi, Guanya},
          journal={arXiv preprint arXiv:2502.01143},
          year={2025}
        }
@misc{HumanoidVerse,
          author = {CMU LeCAR Lab},
          title = {HumanoidVerse: A Multi-Simulator Framework for Humanoid Robot Sim-to-Real Learning},
          year = {2025},
          publisher = {GitHub},
          journal = {GitHub repository},
          howpublished = {\url{https://github.com/LeCAR-Lab/HumanoidVerse}},
        }
```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
