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

        int OW = uOutputSize.x;
        int OH = uOutputSize.y;
        int OC_4 = uOutputSize.z;

        ivec2 s0 = pos.xy*uStride-uPad;
        int kxi, kyi;

        int oc_4 = pos.z;

        ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, uDilate)));
        ivec2 efxy = min(uKernelSize, UP_DIV(inputSize.xy-s0, uDilate));

        vec4 vacc = uBias.data[pos.z];
        vec4 vin, vk;
        ivec4 kBi, inBi;

        for (kyi=sfxy.y; kyi<efxy.y; ++kyi)
        {
            int sy = kyi*uDilate.y + s0.y;
            for (kxi=0; kxi < KW; ++kxi)
            {
                sx = kxi*uDilate.x + s0.x;

                kBi.x = (oc_4+0) * KH * KW + kyi * KW + kxi;
                kBi.y = (oc_4+1) * KH * KW + kyi * KW + kxi;
                kBi.z = (oc_4+2) * KH * KW + kyi * KW + kxi;
                kBi.w = (oc_4+3) * KH * KW + kyi * KW + kxi;

                vk.x = uKernelBuffer[kBi.x];
                vk.y = uKernelBuffer[kBi.y];
                vk.z = uKernelBuffer[kBi.z];
                vk.w = uKernelBuffer[kBi.w];

                inBi.x = (oc_4+0)*H*W + sy*W + sx;
                inBi.y = (oc_4+1)*H*W + sy*W + sx;
                inBi.z = (oc_4+2)*H*W + sy*W + sx;
                inBi.w = (oc_4+3)*H*W + sy*W + sx;

                vin = uInputBuffer[inBi.x];
                vin = uInputBuffer[inBi.x];
                vin = uInputBuffer[inBi.x];
                vin = uInputBuffer[inBi.x];

                vacc += vk * vin;
            }
        }

        int outBi = pos.x + OW*pos.y + 4*pos.z*OW * OH;
        uOutputBuffer.data[outBi+0] = vacc.x;
        uOutputBuffer.data[outBi+1*OWH] = vacc.y;
        uOutputBuffer.data[outBi+2*OWH] = vacc.z;
        uOutputBuffer.data[outBi+3*OWH] = vacc.w;
    }

}
