layout(std430) buffer;
layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(location=1) readonly uniform PRECISION sampler3D uInput;
layout(location=2) uniform ivec3 uInputSize;

layout(binding=3) readonly buffer weight{
  vec4 data[];
} uWeight;

layout(location=4) readonly buffer bias{
  vec4 data[];
} uBias;

layout(location=5) readonly buffer mean{
  vec4 data[];
} uMean;

layout(location=6) readonly buffer variance{
  vec4 data[];
} uVariance;

layout(location=7) uniform float uEps;

layout (local_size_x = WORKGROUP_X, local_size_y = WORKGROUP_Y, local_size_z = WORKGROUP_Z) in;

void main()
{
  ivec3 pos = ivec3(gl_GlobalInvocationID);

  if(all(lessThan(pos, uInputSize.xyz)))
  {
    vec4 color = texelFetch(uInput, ivec3(pos.x, pos.y, pos.z), 0);
    vec4 invVar = inversesqrt(uVariance.data[pos.z] + uEps);
    imageStore(uOutput, pos, (color - uMean.data[pos.z]) * invVar + uBias.data[pos.z]);
  }
}
