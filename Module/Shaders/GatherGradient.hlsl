StructuredBuffer<float> gradientValues : register(t0, space0);
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

uint gathered_source_logical(uint logical, int selected) {
    if (selected < 0 || uint(selected) >= indexHeader.z) return 0xffffffff;
    uint remaining = logical;
    uint result = 0;
    uint sourceStride = 1;
    int indexAxis = int(indexHeader.x) - 1;
    for (int sourceAxis = int(sourceHeader.y) - 1; sourceAxis >= 0; --sourceAxis) {
        if (sourceAxis == int(sourceHeader.z)) {
            result += uint(selected) * sourceStride;
            if (indexHeader.x == sourceHeader.y) {
                uint dimension = pair_component(indexShape0, indexShape1, uint(indexAxis));
                if (dimension > 0) remaining /= dimension;
                --indexAxis;
            }
        } else {
            uint dimension = pair_component(indexShape0, indexShape1, uint(indexAxis));
            uint coordinate = dimension == 0 ? 0 : remaining % dimension;
            if (dimension > 0) remaining /= dimension;
            result += coordinate * sourceStride;
            --indexAxis;
        }
        sourceStride *= pair_component(sourceShape0, sourceShape1, uint(sourceAxis));
    }
    return result;
}

[numthreads(1, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID) {
    uint sourceLogical = id.x;
    uint sourceCount = 1;
    for (uint axis = 0; axis < sourceHeader.y; ++axis) {
        sourceCount *= pair_component(sourceShape0, sourceShape1, axis);
    }
    if (sourceLogical >= sourceCount) return;
    float value = 0.0;
    for (uint logical = 0; logical < sourceHeader.x; ++logical) {
        uint remaining = logical;
        uint indexStorage = indexHeader.y;
        for (int axis = int(indexHeader.x) - 1; axis >= 0; --axis) {
            uint dimension = pair_component(indexShape0, indexShape1, uint(axis));
            uint coordinate = dimension == 0 ? 0 : remaining % dimension;
            if (dimension > 0) remaining /= dimension;
            indexStorage += coordinate * pair_component(indexStrides0, indexStrides1, uint(axis));
        }
        if (gathered_source_logical(logical, indexValues[indexStorage]) == sourceLogical) {
            value += gradientValues[logical];
        }
    }
    outputValues[sourceLogical] = value;
}
