layout(std430) buffer;
layout(binding=0, location=0) writeonly uniform PRECISION image3D uOutput;
layout(binding=1) readonly uniform sampler3D uInput;
layout(binding=2) uniform ivec3 uInputSize;

layout(binding=3) readonly buffer weight{
  vec4 data[];
} uWeight;

layout(binding=4) readonly buffer bias{
  vec4 data[];
} uBias;

layout(binding=5) readonly buffer mean{
  vec4 data[];
} uMean;

layout(binding=6) readonly buffer variance{
  vec4 data[];
} uVariance;

layout(binding=7) uniform float uEps;

layout (local_size_x = COMP_GROUP_X, local_size_y = COMP_GROUP_Y, local_size_z = COMP_GROUP_Z) in;

void main()
{
  ivec3 pos = ivec3(gl_GlobalInvocationID);

  if(all(lessThan(pos, uInputSize.xyz)))
  {
    vec4 color = texelFetch(uInput, ivec3(pos.x, pos.y, pos.z), 0);
    float invVar = inversesqrt(uVariance.data[pos.z] + uEps);
    imageStore(uOutput, pos, (color - uMean.data[pos.z]) * invVar + uBias.data[pos.z]);
  }
}
