# Setup

+ Install dependencies.
+ You might have to run `python3 setup.py develop`
+ Download all model files from [Google Drive](https://drive.google.com/drive/folders/1sxaSIpAh6XOcmWd6dm__96DCamN-lCFX?usp=sharing)
+ Unzip model folders into the `examples/` directory. You should end up with the following model folders: `roberta_ft/`, `uk/` and `ctx/`
+ Setup your custom data in the same way that shown in `undr/`, `fct/` and `both/`. Note that in the latter two folders, the first line is skipped by the file (it can be arbitrary, so I've set it to just be a copy of the second line).

# Running

From the `examples/` directory:

MLM: `sh mlm_scores.sh`

DR-c: `sh dr_c.sh`

DR-f: `sh dr_f.sh`

Regression: `python3 regression.py`

