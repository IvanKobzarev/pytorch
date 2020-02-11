#!/bin/bash
gradle -p android test_app:installBenchMLocalBaseDebug -PABI_FILTERS=arm64-v8a && adb shell am start -n org.pytorch.testapp.benchM/org.pytorch.testapp.MainActivity
