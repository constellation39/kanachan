#/usr/bin/env bash

eval "$(pyenv init -)"
exec tensorboard "$@"
