layout(std430) buffer;
layout(binding=0) writeonly buffer outBuffer{
  float data[];
}uOutBuffer;

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

layout(location=8) uniform ivec4 uOutputSize;
layout(location=9) uniform ivec4 uInputSize;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

layout (local_size_x = WORKGROUP_X, local_size_y = WORKGROUP_Y, local_size_z = WORKGROUP_Z) in;

void main()
{
    if (all(lessThan(ivec3(gl_GlobalInvocationID), uOutputSize.xyz)))
    {
        ivec3 pos = ivec3(gl_GlobalInvocationID)*ivec3(4, 1, 1);
        int KW = uKernelSize.x;
        int KH = uKernelSize.y;
        ivec3 inputSize = uInputSize.xyz;
        int W = uInputSize.x;
        int H = uInputSize.y;
        int C_4 = uInputSize.z;
        int C = uInputSize.w;
        int CAU4 = C_4 * 4;

        int OW = uOutputSize.x;
        int OH = uOutputSize.y;
        int OC_4 = uOutputSize.z;
        int OC = uOutputSize.w;

        ivec2 s0 = pos.xy*uStride-uPad;
        int kxi, kyi, ic_4i;

        int oc_4i = pos.z;

        ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, uDilate)));
        ivec2 efxy = min(uKernelSize, UP_DIV(inputSize.xy-s0, uDilate));

        ivec4 inBi, inBisx0, inBisx1, inBisx2, inBisx3;

        vec4 v[4];
        vec4 invsx[4];

        v[0] = uBias.data[pos.z];
        v[1] = v[0];
        v[2] = v[0];
        v[3] = v[0];
        for (kyi=sfxy.y; kyi<efxy.y; ++kyi)
        {
            int sy = kyi*uDilate.y + s0.y;
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

                int kBi = kBi_oc4i_kyi + CAU4 * kxi;
                for (ic_4i=0; ic_4i < C_4; ++ic_4i)
                {
                    vec4 k0 = uKernelBuffer.data[kBi+0];
                    vec4 k1 = uKernelBuffer.data[kBi+1];
                    vec4 k2 = uKernelBuffer.data[kBi+2];
                    vec4 k3 = uKernelBuffer.data[kBi+3];

                    for (int i = 0; i < 4; ++i) {
                      invsx[i] = vec4(0.0);
                    }
                    int ic4ie = min(4, C - 4*ic_4i);
                    for (int ic4i=0;ic4i<ic4ie; ++ic4i)
                    {
                      int inBi = (4*ic_4i + ic4i)*H*W + sy*W;
                      invsx[0][ic4i] = uInBuffer.data[inBi + sx0];
                      invsx[1][ic4i] = uInBuffer.data[inBi + sx1];
                      invsx[2][ic4i] = uInBuffer.data[inBi + sx2];
                      invsx[3][ic4i] = uInBuffer.data[inBi + sx3];
                    }

                    mat4 k = mat4(k0, k1, k2, k3);
                    v[0] += k * invsx[0] * m0;
                    v[1] += k * invsx[1] * m1;
                    v[2] += k * invsx[2] * m2;
                    v[3] += k * invsx[3] * m3;

                    kBi += 4;
                }
            }
        }

        int vi=0;
        int vie=min(4, OW-pos.x);
        int OWH = OW*OH;
        int outBi;
        for (;vi<vie;++vi)
        {
          outBi = (pos.x+vi) + OW*pos.y + 4*pos.z*OWH;
          vec4 v = v[vi];
          uOutputBuffer.data[outBi+0] = v.x;
          uOutputBuffer.data[outBi+1*OWH] = v.y;
          uOutputBuffer.data[outBi+2*OWH] = v.z;
          uOutputBuffer.data[outBi+3*OWH] = v.w;
        }
    }

}
