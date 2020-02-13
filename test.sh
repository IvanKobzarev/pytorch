echo $ARGS
gradle -PARGS="$ARGS" -p android test_app:installTstLocalBaseDebug -PABI_FILTERS=arm64-v8a && adb shell am start -n org.pytorch.testapp.tst/org.pytorch.testapp.MainActivity
