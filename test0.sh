#!/bin/bash
set -eux

gradle -p android \
  test_app:installTst0LocalBaseDebug \
  -PARGS="" \
  -PABI_FILTERS=arm64-v8a && \
  adb shell am start -n org.pytorch.testapp.tst0/org.pytorch.testapp.MainActivity
