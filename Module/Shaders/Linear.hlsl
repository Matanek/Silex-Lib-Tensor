StructuredBuffer<float> leftValues : register(t0, space0);
StructuredBuffer<float> rightValues : register(t1, space0);
RWStructuredBuffer<float> outputValues : register(u0, space1);

cbuffer Metadata : register(b0, space2) {
    uint4 dimensions;
    uint4 leftLayout;
    uint4 rightLayout;
    uint4 reserved;
};

[numthreads(1, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID) {
    uint outputIndex = id.x;
    if (outputIndex >= dimensions.w) return;
    uint row = outputIndex / dimensions.y;
    uint column = outputIndex % dimensions.y;
    float value = 0.0;
    for (uint inner = 0; inner < dimensions.z; ++inner) {
        uint leftIndex = leftLayout.x + row * leftLayout.z + inner * leftLayout.w;
        uint rightIndex = leftLayout.y + inner * rightLayout.x + column * rightLayout.y;
        value += leftValues[leftIndex] * rightValues[rightIndex];
    }
    outputValues[outputIndex] = value;
}
