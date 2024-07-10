# LOPE
An implementation of Learning Online with trajectory Preference guidancE (LOPE) in PyTorch

## Requirements
Creat the running environment with the following command:
```
conda env create -f environment.yml
```

## How to run it
```
python3 train_lope.py
```

## Noting
In this code, we use the hopper environment to demonstrate the performance improvement of LOPE. You can also adopt the other agents provided in MuJoCo, but the corresponding demonstrations are needed like "hopper_trajs.npy". Moreover, some hyperparameters may need to be adjusted for better evaluation results.

## Citing

Please use this bibtex if you want to cite this repository in your publications :
```
@misc{wang2024preferenceguidedreinforcementlearningefficient,
      title={Preference-Guided Reinforcement Learning for Efficient Exploration}, 
      author={Guojian Wang and Faguo Wu and Xiao Zhang and Tianyuan Chen and Xuyang Chen and Lin Zhao},
      year={2024},
      eprint={2407.06503},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.06503}, 
}
```
