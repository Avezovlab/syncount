#!/usr/bin/zsh

ret=1
while [[ ! $ret -eq 0 ]]; do
    python3 quantif_synapse.py 1 excit
    ret=$?
done
