layout(std430) buffer;
layout(binding=0) writeonly buffer outBuffer{
  vec4 data[];
}uOutputBuffer;

layout(binding=1) readonly buffer inputBuffer{
  vec4 data[];
}uInputBuffer;

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
        ivec3 pos = ivec3(gl_GlobalInvocationID)*ivec3(4, 1, 1);
        int KW = uKernelSize.x;
        int KH = uKernelSize.y;
        ivec3 inputSize = uInputSize;
        int W = uInputSize.x;
        int H = uInputSize.y;
        int C_4 = uInputSize.z;
        int CAU4 = C_4 * 4;

        int OW = uOutputSize.x;
        int OH = uOutputSize.y;
        int OC_4 = uOutputSize.z;

        ivec2 s0 = pos.xy*uStride-uPad;
        int kxi, kyi, ic_4i, oc_4i;
        oc_4i = pos.z;

        ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, uDilate)));
        ivec2 efxy = min(uKernelSize, UP_DIV(inputSize.xy-s0, uDilate));
        vec4 v[4];
        v[0] = uBias.data[pos.z];
        v[1] = v[0];
        v[2] = v[0];
        v[3] = v[0];
        int kBi_oc4i = oc_4i * CAU4 * KH * KW;
        for (kyi=sfxy.y; kyi<efxy.y; ++kyi)
        {
            int sy = kyi*uDilate.y + s0.y;
            int inBi_sy = sy * W;
            int kBi_oc4i_kyi = kBi_oc4i + kyi*KW*CAU4;
            for (kxi=0; kxi < KW; ++kxi)
            {
                int sx0 = kxi*uDilate.x + s0.x;
                int sx1 = sx0 + uStride.x;
                int sx2 = sx1 + uStride.x;
                int sx3 = sx2 + uStride.x;

                float m0 = sx0 >= 0 && sx0 < W ? 1.0 : 0.0;
                float m1 = sx1 >= 0 && sx1 < W ? 1.0 : 0.0;
                float m2 = sx2 >= 0 && sx2 < W ? 1.0 : 0.0;
                float m3 = sx3 >= 0 && sx3 < W ? 1.0 : 0.0;
                int inBi = inBi_sy;
                int kBi = kBi_oc4i_kyi + CAU4 * kxi;
                for (ic_4i=0; ic_4i < C_4; ++ic_4i)
                {
                    vec4 k0 = uKernelBuffer.data[kBi+0];
                    vec4 k1 = uKernelBuffer.data[kBi+1];
                    vec4 k2 = uKernelBuffer.data[kBi+2];
                    vec4 k3 = uKernelBuffer.data[kBi+3];

                    mat4 k = mat4(k0, k1, k2, k3);

                    v[0] += k*uInputBuffer.data[inBi + sx0] * m0;
                    v[1] += k*uInputBuffer.data[inBi + sx1] * m1;
                    v[2] += k*uInputBuffer.data[inBi + sx2] * m2;
                    v[3] += k*uInputBuffer.data[inBi + sx3] * m3;

                    inBi += W*H;
                    kBi += 4;
                }
            }
        }

        int outBi = pos.z * OW * OH + pos.y * OW + pos.x;
        int vi=0;
        int vie=min(4, OW-pos.x);
        for (;vi<vie;++vi)
        {
          uOutputBuffer.data[outBi+vi] = v[vi];
        }
    }

}
