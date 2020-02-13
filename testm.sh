#!/bin/bash

gradle -p android test_app:installTstMLocalBaseDebug -PARGS="" -PABI_FILTERS=arm64-v8a && adb shell am start -n org.pytorch.testapp.tstM/org.pytorch.testapp.MainActivity
