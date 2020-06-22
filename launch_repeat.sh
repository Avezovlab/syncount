#!/usr/bin/zsh


if [ $# -ne 2 ]; then

    echo "Usage: $0 batch [inhib/excit]"
    exit 1
fi

ret=1
while [[ ! $ret -eq 0 ]]; do
    python3 quantif_synapse.py --no-RGB $1 $2
    ret=$?
done
