StructuredBuffer<float> sourceValues : register(t0, space0);
StructuredBuffer<int> indexValues : register(t1, space0);
RWStructuredBuffer<float> outputValues : register(u0, space1);

cbuffer Metadata : register(b0, space2) {
    uint4 sourceHeader;
    uint4 indexHeader;
    uint4 sourceShape0;
    uint4 sourceShape1;
    uint4 sourceStrides0;
    uint4 sourceStrides1;
    uint4 indexShape0;
    uint4 indexShape1;
    uint4 indexStrides0;
    uint4 indexStrides1;
};

uint component(uint4 value, uint index) {
    if (index == 0) return value.x;
    if (index == 1) return value.y;
    if (index == 2) return value.z;
    return value.w;
}

uint pair_component(uint4 first, uint4 second, uint index) {
    if (index < 4) return component(first, index);
    return component(second, index - 4);
}

[numthreads(1, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID) {
    uint logical = id.x;
    if (logical >= sourceHeader.x) return;

    uint remaining = logical;
    uint indexStorage = indexHeader.y;
    uint sourceStorage = sourceHeader.w;
    int indexAxis = int(indexHeader.x) - 1;
    for (int sourceAxis = int(sourceHeader.y) - 1; sourceAxis >= 0; --sourceAxis) {
        if (sourceAxis == int(sourceHeader.z) && indexHeader.x != sourceHeader.y) continue;
        uint dimension = pair_component(indexShape0, indexShape1, uint(indexAxis));
        uint coordinate = dimension == 0 ? 0 : remaining % dimension;
        if (dimension > 0) remaining /= dimension;
        indexStorage += coordinate * pair_component(indexStrides0, indexStrides1, uint(indexAxis));
        if (sourceAxis != int(sourceHeader.z)) {
            sourceStorage += coordinate * pair_component(sourceStrides0, sourceStrides1, uint(sourceAxis));
        }
        --indexAxis;
    }
    int selected = indexValues[indexStorage];
    if (selected < 0 || uint(selected) >= indexHeader.z) {
        outputValues[logical] = asfloat(0x7fc00000);
        return;
    }
    sourceStorage += uint(selected) * pair_component(sourceStrides0, sourceStrides1, sourceHeader.z);
    outputValues[logical] = sourceValues[sourceStorage];
}
