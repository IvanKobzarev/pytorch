#!/bin/bash
GBENCH_REPORT_FILE="agpu-gbench-"$(date "+%m%d-%H%M%S")".json"

if [ -z "$GBENCH_REPORT_FILE" ]
then
  echo "Please set GBENCH_REPORT_FILE"
  exit 1
fi

set -eux -o pipefail
echo "GBENCH_REPORT_FILE:$GBENCH_REPORT_FILE"

gradle -p android test_app:installBenchLocalBaseDebug \
  -PARGS="" \
  -PGBENCH_REPORT_FILE="${GBENCH_REPORT_FILE}" \
  -PABI_FILTERS=armeabi-v7a \
  -PIS=true \
  -PINF=false \
  && adb shell am start -n org.pytorch.testapp.bench/org.pytorch.testapp.MainActivity

echo "\n\n\nGBENCH_REPORT_FILE:$GBENCH_REPORT_FILE"


echo "Probably next command:"
echo "GBENCH_REPORT_FILE=$GBENCH_REPORT_FILE sh ./bench_reportfile_process.sh"

