package org.pytorch.testapp;

import android.app.admin.DevicePolicyManager;
import android.content.Context;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.HardwarePropertiesManager;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.annotation.WorkerThread;
import androidx.appcompat.app.AppCompatActivity;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

  private static final String TAG = BuildConfig.LOGCAT_TAG;
  private static final int TEXT_TRIM_SIZE = 4096;

  private TextView mTextView;
  private View mStartInteractiveButton;

  protected HandlerThread mBackgroundThread;
  protected Handler mBackgroundHandler;

  protected HandlerThread mLogThread;
  protected Handler mLogHandler;
  protected Handler mUiHandler = new Handler(Looper.getMainLooper());

  private Module mModule;
  private FloatBuffer mInputTensorBuffer;
  private Tensor mInputTensor;
  private StringBuilder mTextViewStringBuilder = new StringBuilder();

  private final Runnable mLogRunnable = new Runnable() {
    @Override
    public void run() {
      if (Build.VERSION.SDK_INT >= 24) {
        final DevicePolicyManager dpm = (DevicePolicyManager) getSystemService(Context.DEVICE_POLICY_SERVICE);
        final String pkgName = MainActivity.this.getPackageName();
        boolean isDeviceOwnerApp = dpm.isDeviceOwnerApp(pkgName);
        Log.i(TAG, String.format("IIILog isDeviceOwnerApp:%b", isDeviceOwnerApp));
        if (isDeviceOwnerApp) {
          try {
            HardwarePropertiesManager hwpm = (HardwarePropertiesManager) getSystemService(MainActivity.this.HARDWARE_PROPERTIES_SERVICE);
            int typeGpu = HardwarePropertiesManager.DEVICE_TEMPERATURE_GPU;
            int typeCpu = HardwarePropertiesManager.DEVICE_TEMPERATURE_CPU;

            int srcCurr = HardwarePropertiesManager.TEMPERATURE_CURRENT;
            int srcThrtl = HardwarePropertiesManager.TEMPERATURE_THROTTLING;

            float[] gpuCurr = hwpm.getDeviceTemperatures(typeGpu, srcCurr);
            float[] gpuThrtl = hwpm.getDeviceTemperatures(typeGpu, srcThrtl);
            Log.i(TAG, String.format("IIILog GPU temps curr:%s thrtl:%s",
                Arrays.toString(gpuCurr), Arrays.toString(gpuThrtl)));

            float[] cpuCurr = hwpm.getDeviceTemperatures(typeGpu, srcCurr);
            float[] cpuThrtl = hwpm.getDeviceTemperatures(typeCpu, srcThrtl);
            Log.i(TAG, String.format("IIILog CPU temps curr:%s thrtl:%s",
                Arrays.toString(cpuCurr), Arrays.toString(cpuThrtl)));
          } catch(Exception e) {
            Log.e(TAG, "Error get hw temp", e);
          }

          if (mLogHandler != null) {
            mLogHandler.postDelayed(mLogRunnable, 5000);
          }
        }
      }
    }
  };

  private final Runnable mModuleForwardRunnable = new Runnable() {
    @Override
    public void run() {
      final String brand = android.os.Build.BRAND;
      final String model = android.os.Build.MODEL;
      final int sdkInt = Build.VERSION.SDK_INT;
      final String abis = Arrays.toString(Build.SUPPORTED_ABIS);
      StringBuilder osbSb = new StringBuilder();
      osbSb
          .append(brand)
          .append(' ')
          .append(model)
          .append(' ')
          .append(sdkInt)
          .append(' ')
          .append(abis);
      String osBuildInfo = osbSb.toString();

      Log.i(TAG, "III android.os.Build:" + osBuildInfo);

      Log.i(TAG, "BuildConfig.AGPU_TEST0:" + BuildConfig.AGPU_TEST0);
      Log.i(TAG, "BuildConfig.AGPU_GTEST:" + BuildConfig.AGPU_GTEST);
      Log.i(TAG, "BuildConfig.AGPU_GBENCH:" + BuildConfig.AGPU_GBENCH);
      Log.i(TAG, "BuildConfig.AGPU_GTEST_M:" + BuildConfig.AGPU_GTEST_M);
      Log.i(TAG, "BuildConfig.AGPU_GBENCH_M:" + BuildConfig.AGPU_GBENCH_M);
      if (BuildConfig.AGPU_GTEST_M != null) {
        PyTorchAndroid.nativeAgpuGTestModule(BuildConfig.AGPU_GTEST_M, getAssets(), BuildConfig.AGPU_GTEST);
      } else if (BuildConfig.AGPU_GBENCH_M != null) {
        PyTorchAndroid.nativeAgpuGBenchModule(BuildConfig.AGPU_GBENCH_M, getAssets(), BuildConfig.AGPU_GBENCH);
      } else if (BuildConfig.AGPU_TEST0 != null) {
        PyTorchAndroid.nativeAgpuTest0(BuildConfig.AGPU_TEST0);
      } else if (BuildConfig.AGPU_GTEST != null) {
        PyTorchAndroid.nativeAgpuGTest(BuildConfig.AGPU_GTEST);
      } else if (BuildConfig.AGPU_GBENCH != null) {


        int i = 0;
        Log.i(TAG, "BuildConfig.AGPU_GBENCH_INF:" + BuildConfig.AGPU_GBENCH_INF);
        int iLimit = BuildConfig.AGPU_GBENCH_INF
            ? Integer.MAX_VALUE
            : 1;
        while (i < iLimit) {
          String reportFileName = BuildConfig.AGPU_GBENCH_REPORT_FILE;
          if (i > 0) {
            reportFileName += Integer.toString(i);
          }
          Log.i(TAG, "III AGPU_GBENCH reportFileName:" + reportFileName);
          final File file = new File(getExternalFilesDir(null), reportFileName);
          final String absPath = file.getAbsolutePath();
          Log.i(TAG, "III AGPU_GBENCH reportFile absPath:" + absPath);

          StringBuilder sb = new StringBuilder();
          sb
              .append(BuildConfig.AGPU_GBENCH)
              .append("--benchmark_out=")
              .append(absPath)
              .append(" --benchmark_out_format=json");
          final String args = sb.toString();

          Log.i(TAG, "III AGPU_GBENCH args:" + args);

          PyTorchAndroid.nativeAgpuGBench(args, osBuildInfo);

          for (int _i = 0; _i < 10000; _i++) {
            Log.i(TAG, "III AGPU_GBENCH COMPLETED " + i + " " + reportFileName);
          }
          i++;
        }
        mUiHandler.post(new Runnable() {
          @Override
          public void run() {
            mTextView.setBackgroundColor(Color.RED);
          }
        });
      } else {
        final Result result = doModuleForward();
        runOnUiThread(new Runnable() {
          @Override
          public void run() {
            handleResult(result);
            //if (mBackgroundHandler != null) {
            //  mBackgroundHandler.post(mModuleForwardRunnable);
            //}
          }
        });
      }
    }
  };
  private View mStartPreburnButton;
  private boolean mPreburnStarted = false;

  private void gbench_run(String absFilePath, String info) {

    Log.i(TAG, "III AGPU_GBENCH reportFile absPath:" + absFilePath);
    StringBuilder sb = new StringBuilder();
    sb.append(BuildConfig.AGPU_GBENCH);

    if (absFilePath != null) {
        sb.append("--benchmark_out=")
          .append(absFilePath)
          .append(" --benchmark_out_format=json");
    }
    final String args = sb.toString();
    Log.i(TAG, "III AGPU_GBENCH args:" + args);

    PyTorchAndroid.nativeAgpuGBench(args, info);
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    mTextView = findViewById(R.id.text);
    mStartInteractiveButton = findViewById(R.id.start_interactive);
    mStartPreburnButton = findViewById(R.id.start_preburn);
    startBackgroundThread();

    Log.i(TAG, "BuildConfig.AGPU_GBENCH_INTERACTIVE:" + BuildConfig.AGPU_GBENCH_INTERACTIVE);

    if (BuildConfig.AGPU_GBENCH_INTERACTIVE) {
      mStartPreburnButton.setVisibility(View.VISIBLE);
      mStartInteractiveButton.setVisibility(View.VISIBLE);
      mStartPreburnButton.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
          if (mPreburnStarted) {
            return;
          }
          mPreburnStarted = true;

          mBackgroundHandler.post(new Runnable() {
            @Override
            public void run() {
              String reportFileName = BuildConfig.AGPU_GBENCH_REPORT_FILE+"-preburn";
              final File file = new File(getExternalFilesDir(null), reportFileName);
              final String absPath = file.getAbsolutePath();
              gbench_run(absPath, "");
              for (int _i = 0; _i < 10000; _i++) {
                Log.i(TAG, "III AGPU_GBENCH PREBURN COMPLETED " + reportFileName);
              }
              mUiHandler.post(new Runnable() {
                @Override
                public void run() {
                  mStartPreburnButton.setBackgroundColor(Color.GREEN);
                }
              });
            }
          });

          mStartPreburnButton.setBackgroundColor(Color.GRAY);
          mStartPreburnButton.setEnabled(false);
        }
      });
      mStartInteractiveButton.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
          mStartInteractiveButton.setVisibility(View.INVISIBLE);
          mBackgroundHandler.post(mModuleForwardRunnable);
        }
      });
    } else {
      mBackgroundHandler.post(mModuleForwardRunnable);
    }
    mLogHandler.post(mLogRunnable);
  }

  protected void startBackgroundThread() {
    mBackgroundThread = new HandlerThread(TAG + "_bg");
    mBackgroundThread.start();
    mBackgroundHandler = new Handler(mBackgroundThread.getLooper());

    mLogThread = new HandlerThread(TAG + "_log");
    mLogThread.start();
    mLogHandler = new Handler(mLogThread.getLooper());
  }

  @Override
  protected void onDestroy() {
    stopBackgroundThread();
    super.onDestroy();
  }

  protected void stopBackgroundThread() {
    mBackgroundThread.quitSafely();
    try {
      mBackgroundThread.join();
      mBackgroundThread = null;
      mBackgroundHandler = null;
    } catch (InterruptedException e) {
      Log.e(TAG, "Error stopping background thread", e);
    }

    mLogThread.quitSafely();
    try {
      mLogThread.join();
      mLogThread = null;
      mLogHandler = null;
    } catch (InterruptedException e) {
      Log.e(TAG, "Error stopping log thread", e);
    }
  }

  @WorkerThread
  @Nullable
  protected Result doModuleForward() {
    if (mModule == null) {
      final long[] shape = BuildConfig.INPUT_TENSOR_SHAPE;
      long numElements = 1;
      for (int i = 0; i < shape.length; i++) {
        numElements *= shape[i];
      }
      mInputTensorBuffer = Tensor.allocateFloatBuffer((int) numElements);
      mInputTensor = Tensor.fromBlob(mInputTensorBuffer, BuildConfig.INPUT_TENSOR_SHAPE);
      //PyTorchAndroid.setNumThreads(1);
      mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), BuildConfig.MODULE_ASSET_NAME);
    }

    final long startTime = SystemClock.elapsedRealtime();
    final long moduleForwardStartTime = SystemClock.elapsedRealtime();
    final Tensor outputTensor = mModule.forward(IValue.from(mInputTensor)).toTensor();
    final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;
    final float[] scores = outputTensor.getDataAsFloatArray();
    final long analysisDuration = SystemClock.elapsedRealtime() - startTime;

    return new Result(scores, moduleForwardDuration, analysisDuration);
  }

  static class Result {

    private final float[] scores;
    private final long totalDuration;
    private final long moduleForwardDuration;

    public Result(float[] scores, long moduleForwardDuration, long totalDuration) {
      this.scores = scores;
      this.moduleForwardDuration = moduleForwardDuration;
      this.totalDuration = totalDuration;
    }
  }

  @UiThread
  protected void handleResult(Result result) {
    String message = String.format("forwardDuration:%d", result.moduleForwardDuration);
    Log.i(TAG, message);
    mTextViewStringBuilder.insert(0, '\n').insert(0, message);
    if (mTextViewStringBuilder.length() > TEXT_TRIM_SIZE) {
      mTextViewStringBuilder.delete(TEXT_TRIM_SIZE, mTextViewStringBuilder.length());
    }
    mTextView.setText(mTextViewStringBuilder.toString());
  }
}
