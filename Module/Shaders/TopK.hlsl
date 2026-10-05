StructuredBuffer<float> inputValues : register(t0, space0);
RWStructuredBuffer<int> outputIndices : register(u0, space1);

cbuffer Metadata : register(b0, space2) {
    uint4 header0; // largest, axis, rank, groups
    uint4 header1; // domain, offset, k, inner
    uint4 shape0;
    uint4 shape1;
    uint4 strides0;
    uint4 strides1;
};

uint component(uint4 value, uint index) {
    if (index == 0u) return value.x;
    if (index == 1u) return value.y;
    if (index == 2u) return value.z;
    return value.w;
}
uint shape_at(uint axis) { return axis < 4u ? component(shape0,axis) : component(shape1,axis-4u); }
uint stride_at(uint axis) { return axis < 4u ? component(strides0,axis) : component(strides1,axis-4u); }

bool precedes(float left, uint leftIndex, float right, uint rightIndex) {
    if (isnan(left)) return !isnan(right) || leftIndex < rightIndex;
    if (isnan(right)) return false;
    if (left == right) return leftIndex < rightIndex;
    return header0.x != 0u ? left > right : left < right;
}

[numthreads(1, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID) {
    uint group = id.x;
    if (group >= header0.w) return;
    uint base = header1.y;
    uint remaining = group;
    for (int axis = int(header0.z)-1; axis >= 0; --axis) {
        if (uint(axis) == header0.y) continue;
        uint dimension = shape_at(uint(axis));
        uint coordinate = remaining % dimension;
        remaining /= dimension;
        base += coordinate * stride_at(uint(axis));
    }
    float previousValue = 0.0;
    uint previousIndex = 0u;
    for (uint rank = 0u; rank < header1.z; ++rank) {
        bool found = false;
        float bestValue = 0.0;
        uint bestIndex = 0u;
        for (uint term = 0u; term < header1.x; ++term) {
            float value = inputValues[base + term*stride_at(header0.y)];
            bool eligible = rank == 0u || precedes(previousValue,previousIndex,value,term);
            if (eligible && (!found || precedes(value,term,bestValue,bestIndex))) {
                found = true;
                bestValue = value;
                bestIndex = term;
            }
        }
        uint outputIndex = (group/header1.w)*header1.z*header1.w + rank*header1.w + group%header1.w;
        outputIndices[outputIndex] = int(bestIndex);
        previousValue = bestValue;
        previousIndex = bestIndex;
    }
}
