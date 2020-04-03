layout(std430) buffer;
layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(location=1) uniform PRECISION sampler3D uM1;
layout(location=2) uniform PRECISION sampler3D uM2;
layout(location=3) uniform PRECISION sampler3D uT;

layout(location=4) uniform float uBeta;
layout(location=5) uniform float uAlpha;

layout(location=6) uniform ivec3 uOutputSize;
layout(location=7) uniform int uK;

layout (local_size_x = WORKGROUP_X, local_size_y = WORKGROUP_Y, local_size_z = WORKGROUP_Z) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (all(lessThan(pos, uOutputSize)))
    {
      int K = uK;
      vec4 mmv = vec4(0);
      int ki = 0;
      for (; ki<K; ++ki)
      {
        vec4 m1ki = texelFetch(uM1, ivec3(ki, pos.y, pos.z), 0);
        vec4 m2ki = texelFetch(uM2, ivec3(pos.x, ki, pos.z), 0);
        mmv += m1ki * m2ki;
      }

      vec4 tv = texelFetch(uT, pos, 0);
      imageStore(uOutput, pos, uBeta * tv + uAlpha * mmv);
    }
}
