# DRP Pre-release

This repository contains the pre-release version of DRP: Deep Reactive Policy, with the IMPACT model and checkpoints.

## Installation

```bash
# Create the conda environment
conda create -n drp python=3.8
conda activate drp

# Clone and install DRP
git clone -b v0.0.0 git@github.com:deep-reactive-policy/drp.git
cd drp
pip install -e .
pip install -r requirements.txt

# Install robofin
git clone -b v0.0.1 git@github.com:Jim-Young6709/robofin.git
pip install -e robofin/

# Install pointnet2_ops. This step may take a few minutes
git clone -b drp git@github.com:Jim-Young6709/pointnet2_ops.git
pip install -e pointnet2_ops/ --no-build-isolation
```

## Test the Installation

Run the example inference script, which launches an interactive DRP GUI:

```bash
python drp/drp_inference.py
```

https://github.com/user-attachments/assets/1074ddb1-2c5c-4470-bf86-3023c83b84ea

## Citation
If you find this codebase useful in your research, please cite:
```bibtex
@article{yang2025deep,
  title={Deep Reactive Policy: Learning Reactive Manipulator Motion Planning for Dynamic Environments},
  author={Jiahui Yang and Jason Jingzhou Liu and Yulong Li and Youssef Khaky and Deepak Pathak},
  journal={9th Annual Conference on Robot Learning},
  year={2025},
}

@inproceedings{dalal2025neural,
  title={Neural MP: A Neural Motion Planner},
  author={Dalal, Murtaza and Yang, Jiahui and Mendonca, Russell and Khaky, Youssef and Salakhutdinov, Ruslan and Pathak, Deepak},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={1--8},
  year={2025},
  organization={IEEE}
}
```
