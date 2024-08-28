#/usr/bin/env bash

eval "$(pyenv init -)"
exec python3 -m kanachan.simulation.run "$@"
