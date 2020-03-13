package org.pytorch.testapp;

import android.app.admin.DeviceAdminReceiver;
import android.app.admin.DevicePolicyManager;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;

public class DeviceOwnerReceiver extends DeviceAdminReceiver {
  @Override
  public void onProfileProvisioningComplete(Context context, Intent intent) {
    DevicePolicyManager manager =
        (DevicePolicyManager) context.getSystemService(Context.DEVICE_POLICY_SERVICE);
    ComponentName componentName = getComponentName(context);
    manager.setProfileName(componentName, context.getString(R.string.profile_name));

    Intent launch = new Intent(context, MainActivity.class);
    launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
    context.startActivity(launch);
  }

  public static ComponentName getComponentName(Context context) {
    return new ComponentName(context.getApplicationContext(), DeviceOwnerReceiver.class);
  }
}
