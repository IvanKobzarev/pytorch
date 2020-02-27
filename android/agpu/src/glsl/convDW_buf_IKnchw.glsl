layout(std430) buffer;
layout(binding=0) writeonly buffer outBuffer{
  float data[];
}uOutputBuffer;

layout(binding=1) readonly buffer inputBuffer{
  float data[];
}uInBuffer;

layout(binding=2) readonly buffer kernelBuffer{
  float data[];
}uKernelBuffer;

layout(binding=3) readonly buffer bias{
  float data[];
}uBias;

layout(location=4) uniform ivec2 uPad;
layout(location=5) uniform ivec2 uKernelSize;
layout(location=6) uniform ivec2 uStride;
layout(location=7) uniform ivec2 uDilate;

layout(location=8) uniform ivec3 uOutputSize;
layout(location=9) uniform ivec3 uInputSize;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

layout (local_size_x = WORKGROUP_X, local_size_y = WORKGROUP_Y, local_size_z = WORKGROUP_Z) in;

void main()
{
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(ivec3(gl_GlobalInvocationID), uOutputSize)))
  {
    int KW = uKernelSize.x;
    int KH = uKernelSize.y;
    ivec3 inputSize = uInputSize;
    int W = uInputSize.x;
    int H = uInputSize.y;
    int C = uInputSize.z;

    int OW = uOutputSize.x;
    int OH = uOutputSize.y;
    int GOC = uOutputSize.z;

    ivec2 s0 = pos.xy*uStride-uPad;
    int sx;
    int kxi, kyi;

    int goc = pos.z;
    int oci = goc / C;
    int ici = goc % C;

    ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, uDilate)));
    ivec2 efxy = min(uKernelSize, UP_DIV(inputSize.xy-s0, uDilate));

    float acc = uBias.data[oci];

    for (kyi=sfxy.y; kyi<efxy.y; ++kyi)
    {
      int sy = kyi*uDilate.y + s0.y;
      for (kxi=0; kxi < KW; ++kxi)
      {
        sx = kxi*uDilate.x + s0.x;
        acc += uKernelBuffer.data[goc*KH*KW + kyi*KW + kxi] * uInBuffer.data[ici*H*W + sy*W + sx];
      }
    }

    uOutputBuffer.data[pos.x + OW*pos.y + goc*OW*OH] = acc;
  }
}
