
layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(location=1) uniform mediump sampler3D uInput;
layout(location=2) uniform ivec3 uImgSize;
layout(location=3) uniform float uThreshold;
layout(location=4) uniform float uValue;

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if(pos.x < uImgSize.x && pos.y < uImgSize.y && pos.z < uImgSize.z)
    {
      vec4 dataIn = texelFetch(uInput, pos, 0);
      bvec4 lessThreshold = bvec4(lessThan(dataIn, vec4(uThreshold)));
      imageStore(uOutput, pos, mix(dataIn, uValue, lessThreshold));
    }
}
