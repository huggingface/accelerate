# Config Zoo

This folder contains a variety of minimal configurations for `Accelerate` achieving certain goals. You can use these 
direct config YAML's, or build off of them for your own YAML's.

These are highly annoted versions, aiming to teach you what each section does.

Each config can be run via `accelerate launch --config_file {file} run_me.py`

`run_me.py` will then print out how the current environment is setup (the contents of the `AcceleratorState`)