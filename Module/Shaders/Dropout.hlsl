StructuredBuffer<float> inputValues : register(t0, space0);
RWStructuredBuffer<float> outputValues : register(u0, space1);

cbuffer Metadata : register(b0, space2) {
    uint4 header;
    uint4 shape0;
    uint4 shape1;
    uint4 strides0;
    uint4 strides1;
};

cbuffer Probability : register(b1, space2) { float probability; };
cbuffer Scale : register(b2, space2) { float scale; };
cbuffer Counter : register(b3, space2) { uint seed; uint iteration; };

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

uint storage_index(uint logical) {
    uint result = header.w;
    uint remaining = logical;
    for (int axis = int(header.z) - 1; axis >= 0; --axis) {
        uint dimension = pair_component(shape0, shape1, uint(axis));
        uint coordinate = dimension == 0 ? 0 : remaining % dimension;
        if (dimension > 0) remaining /= dimension;
        result += coordinate * pair_component(strides0, strides1, uint(axis));
    }
    return result;
}

uint random_bits(uint index) {
    uint value = seed ^ (iteration * 0x9E3779B9u) ^ (index + 1u) * 0x85EBCA6Bu;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
}

[numthreads(1, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID) {
    uint logical = id.x;
    if (logical >= header.y) return;
    float sample = float(random_bits(logical) & 0x00FFFFFFu) / 16777216.0;
    outputValues[logical] = sample < probability ? 0.0 : inputValues[storage_index(logical)] * scale;
}
