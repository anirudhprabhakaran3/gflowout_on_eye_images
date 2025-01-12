# Enhance Eye Disease Detection using Learnable Probabilistic Discrete Latents in Machine Learning Architectures

This is the official code repository for the project "Enhance Eye Disease Detection using Learnable Probabilistic Discrete Latents in Machine Learning Architectures". You can find the pre-print of the paper [here](https://arxiv.org/abs/2402.16865).

This work is adapted from [GFNDropout](https://github.com/kaiyuanmifen/GFNDropout).

## Steps to run
We recommend using a virtual environment to run this project. Also, a system with a GPU is recommended.

1. Install all the dependencies

```
pip install -r requirements.txt
```

2. Run the experiment script

```
python rfmid_gfn.py
```

You can make changes to the experiment settings by changing this master script. New datasets can be added by adding corresponding PyTorch classes in the `data` folder and exporting them for use. Along with this, `Options` for running the project can be edited in either the main class, located at `utils/options`, or can be set while the object is being initialized.