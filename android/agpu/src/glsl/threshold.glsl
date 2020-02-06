layout(FORMAT, binding=0, location=0) writeonly uniform PRECISION image3D uOutput;
layout(FORMAT, binding=1, location=1) readonly uniform PRECISION sampler3D uInput;
layout(location=2) uniform ivec4 uImgSize;
layout(location=3) uniform float uThreshold;
layout(location=4) uniform float uValue;

layout (local_size_x = COMP_GROUP_X, local_size_y = COMP_GROUP_Y, local_size_z = COMP_GROUP_Z) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 imgSize = uImgSize.xyz;
    if(pos.x < imgSize.x && pos.y < imgSize.y && pos.z < imgSize.z)
    {
      vec4 dataIn = texelFetch(uInput, pos, 0);
      bvec4 lessThreshold = bvec4(lessThan(dataIn, vec4(uThreshold)));
      imageStore(uOutput, pos, mix(dataIn, vec4(uValue), lessThreshold));
    }
}
