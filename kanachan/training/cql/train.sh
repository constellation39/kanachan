#/usr/bin/env bash

eval "$(pyenv init -)"
exec torchrun --nproc_per_node gpu --standalone -m kanachan.training.cql.train "$@"
