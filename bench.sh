gradle -p android test_app:installBenchLocalBaseDebug \
  -PARGS="" \
  -PABI_FILTERS=arm64-v8a && 
adb shell am start -n org.pytorch.testapp.bench/org.pytorch.testapp.MainActivity
