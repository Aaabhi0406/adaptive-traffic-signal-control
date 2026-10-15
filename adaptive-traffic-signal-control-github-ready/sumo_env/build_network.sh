#!/bin/bash
# Reproducible network build for the 4-way intersection used by
# sumo_integration.py. Regenerate with: bash build_network.sh
set -e
export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
netgenerate --spider --spider.arm-number=4 --spider.circle-number=1 --spider.space-radius=150 \
  --default.lanenumber=1 --no-turnarounds --tls.set=A1 --tls.default-type=static \
  --output-file=intersection.net.xml
echo "Built intersection.net.xml"
