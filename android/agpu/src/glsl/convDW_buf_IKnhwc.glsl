layout(std430) buffer;
layout(binding=0) writeonly buffer outBuffer{
  float data[];
}uOutputBuffer;

layout(binding=1) readonly buffer inputBuffer{
  float data[];
}uInBuffer;

layout(binding=2) readonly buffer kernelBuffer{
  vec4 data[];
}uKernelBuffer;

layout(binding=3) readonly buffer bias{
  vec4 data[];
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
    if (all(lessThan(ivec3(gl_GlobalInvocationID), uOutputSize)))
    {
        ivec3 pos = ivec3(gl_GlobalInvocationID);
        int KW = uKernelSize.x;
        int KH = uKernelSize.y;
        ivec3 inputSize = uInputSize;
        int W = uInputSize.x;
        int H = uInputSize.y;
        int C = uInputSize.y;

        int OW = uOutputSize.x;
        int OH = uOutputSize.y;
        int C_4 = uOutputSize.z;

        ivec2 s0 = pos.xy*uStride-uPad;
        int kxi, kyi;

        int c_4 = pos.z;

        ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, uDilate)));
        ivec2 efxy = min(uKernelSize, UP_DIV(inputSize.xy-s0, uDilate));

        vec4 vacc = uBias.data[pos.z];
        vec4 vin, vk;
        ivec4 kBi, inBi;

        int c_4_ie = min(4, C - 4*c_4);
        int c_4_i;
        int inBi;
        int kBi;

        for (kyi=sfxy.y; kyi<efxy.y; ++kyi)
        {
            int sy = kyi*uDilate.y + s0.y;
            for (kxi=0; kxi < KW; ++kxi)
            {
                sx = kxi*uDilate.x + s0.x;
                vk = {0.0, 0.0, 0.0, 0.0};
                for(c_4_i=0; c_4_i<c_4_ie; ++c_4_i)
                {
                  inBi = (oc_4+c_4_i) + (sy*W + sx) * C;
                  kBi = (oc_4+c_4_i) + (kyi*KW + kxi) * OC;
                  vk[c_4_i] = uKernelBuffer[kBi] * uInputBuffer[inBi];
                }

                vacc += vk * vin;
            }
        }

        int outBi = OC*(OW*pos.y + pos.x) + 4*pos.z;
        for(c_4_i=0; c_4_i<c_4_ie; ++c_4_i)
        {
            uOutputBuffer.data[outBi + c_4_i] = vacc[c_4_i];
        }
    }

}
