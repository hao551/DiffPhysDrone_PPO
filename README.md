# Vision-based Agile Flight Training Code

## Overview

This repository contains the training code for our research on **Learning Vision-based Agile Flight via Differentiable Physics**. Accepted by [**Nature Machine Intelligence'25**](https://www.nature.com/articles/s42256-025-01048-0). 

[Project webpage](https://henryhuyu.github.io/DiffPhysDrone_Web/) is alive.

## Quick Demos
### Single Agent Flights
<table>
  <tr>
    <td><img src="./gifs/20ms.gif" alt="GIF 1" width="300"></td>
    <td><img src="./gifs/fpv_dense.gif" alt="GIF 2" width="300"></td>
  </tr>
</table>

### Swarm Tasks
<table>
  <tr>
    <td><img src="./gifs/swap_position.gif" alt="GIF 1" width="300"></td>
    <td><img src="./gifs/main_task.gif" alt="GIF 2" width="300"></td>
  </tr>
</table>


## Environment Setup
### Python Environment

The code is tested with the following environment:

- **PyTorch**: 2.2.2
- **Python**: 3.11
- **CUDA**: 11.8

The code should be compatible with other PyTorch and CUDA versions.

### Build CUDA Ops

To build the CUDA operations, run the following command:

```bash
pip install -e src
```

## Training

To start the training process, use the following command:

```bash
# For multi-agemt
python main_cuda.py $(cat configs/multi_agent.args)
# For single-agemt
python main_cuda.py $(cat configs/single_agent.args)
```

## Evaluation
You need to download the simulation validation code from the GitHub release page.
To evaluate the trained model in multi-agent settings, use the following command to launch the simulator:
```bash
cd <path to multi agent code supplementary>
./LinuxNoEditor/Blocks.sh -ResX=896 -ResY=504 -windowed -WinX=512 -WinY=304 -settings=$PWD/settings.json
```

## Citation
If using this repository, please cite our work
```
@article{zhang2025learning,
  title={Learning vision-based agile flight via differentiable physics},
  author={Zhang, Yuang and Hu, Yu and Song, Yunlong and Zou, Danping and Lin, Weiyao},
  journal={Nature Machine Intelligence},
  pages={1--13},
  year={2025},
  publisher={Nature Publishing Group}
}
```
Then, run the following command to evaluate the trained model:
```bash
python eval.py --resume <path to checkpoint> --target_speed 2.5
```
