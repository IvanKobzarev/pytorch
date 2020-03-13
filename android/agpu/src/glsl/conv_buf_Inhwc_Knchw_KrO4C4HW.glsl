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
        ivec3 pos = ivec3(gl_GlobalInvocationID)*ivec3(4, 1, 1);
        int KW = uKernelSize.x;
        int KH = uKernelSize.y;
        ivec3 inputSize = uInputSize;
        int W = uInputSize.x;
        int H = uInputSize.y;
        int C = uInputSize.z;
        int C_4 = UP_DIV(C, 4);

        int OW = uOutputSize.x;
        int OH = uOutputSize.y;
        int OC = uOutputSize.z;
        int OC_4 = UP_DIV(OC, 4);

        ivec2 s0 = pos.xy*uStride-uPad;
        int kxi, kyi, ic_4i;

        int oc_4i = pos.z;

        ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, uDilate)));
        ivec2 efxy = min(uKernelSize, UP_DIV(inputSize.xy-s0, uDilate));
        vec4 v[4];
        v[0] = uBias.data[pos.z];
        v[1] = v[0];
        v[2] = v[0];
        v[3] = v[0];
        ivec4 inBi, inBisx0, inBisx1, inBisx2, inBisx3;
        vec4 k0, k1, k2, k3;
        int kBi_oc4_i = oc_4i * (C_4 * KH * KW * 4);
        int kBi;
        int sx0, sx1, sx2, sx3;
        for (kyi=sfxy.y; kyi<efxy.y; ++kyi)
        {
            int sy = kyi*uDilate.y + s0.y;
            for (kxi=0; kxi < KW; ++kxi)
            {
                sx0 = kxi*uDilate.x + s0.x;
                sx1 = sx0 + uStride.x;
                sx2 = sx1 + uStride.x;
                sx3 = sx2 + uStride.x;

                float m0 = sx0 >= 0 && sx0 < W ? 1.0 : 0.0;
                float m1 = sx1 >= 0 && sx1 < W ? 1.0 : 0.0;
                float m2 = sx2 >= 0 && sx2 < W ? 1.0 : 0.0;
                float m3 = sx3 >= 0 && sx3 < W ? 1.0 : 0.0;

                kBi = kBi_oc4_i + (kxi + kyi*KW) * 4;

                for (ic_4i=0; ic_4i < C_4; ++ic_4i)
                {
                    k0 = uKernelBuffer.data[kBi+0];
                    k1 = uKernelBuffer.data[kBi+1];
                    k2 = uKernelBuffer.data[kBi+2];
                    k3 = uKernelBuffer.data[kBi+3];

                    mat4 k = mat4(k0, k1, k2, k3);

                    inBi.x = W * C * sy + ic_4i + 0;
                    inBi.y = W * C * sy + ic_4i + 1;
                    inBi.z = W * C * sy + ic_4i + 2;
                    inBi.w = W * C * sy + ic_4i + 3;

                    inBisx0 = inBi + C * sx0;
                    inBisx1 = inBi + C * sx1;
                    inBisx2 = inBi + C * sx2;
                    inBisx3 = inBi + C * sx3;

                    vec4 invsx0 = vec4(uInBuffer.data[inBisx0.x],uInBuffer.data[inBisx0.y],uInBuffer.data[inBisx0.z],uInBuffer.data[inBisx0.w]);
                    vec4 invsx1 = vec4(uInBuffer.data[inBisx1.x],uInBuffer.data[inBisx1.y],uInBuffer.data[inBisx1.z],uInBuffer.data[inBisx1.w]);
                    vec4 invsx2 = vec4(uInBuffer.data[inBisx2.x],uInBuffer.data[inBisx2.y],uInBuffer.data[inBisx2.z],uInBuffer.data[inBisx2.w]);
                    vec4 invsx3 = vec4(uInBuffer.data[inBisx3.x],uInBuffer.data[inBisx3.y],uInBuffer.data[inBisx3.z],uInBuffer.data[inBisx3.w]);

                    v[0] += k * invsx0 * m0;
                    v[1] += k * invsx1 * m1;
                    v[2] += k * invsx2 * m2;
                    v[3] += k * invsx3 * m3;
                    kBi += KH * KW * 4;
                }
            }
        }

        int vxi = 0;
        int vxie = min(4, OW-pos.x);

        int vci = 0;
        int vcie = min(4, OC-4*pos.z);
        for (;vxi<vxie;++vxi)
        {
          int outBi = OW*OC*pos.y + OC*(pos.x + vxi) + 4*pos.z;
          vec4 v = v[vxi];
          vci = 0;
          for(;vci<vcie;++vci)
          {
            uOutputBuffer.data[outBi + vci] = v[vci];
          }
        }
    }

}
