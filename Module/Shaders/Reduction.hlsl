StructuredBuffer<float> inputValues : register(t0, space0);
RWStructuredBuffer<float> outputValues : register(u0, space1);

cbuffer Metadata : register(b0, space2) {
    uint4 header0;
    uint4 header1;
    uint4 shape0;
    uint4 shape1;
    uint4 strides0;
    uint4 strides1;
    uint4 reduced0;
    uint4 reduced1;
};

uint component(uint4 value, uint index) {
    if (index == 0) return value.x;
    if (index == 1) return value.y;
    if (index == 2) return value.z;
    return value.w;
}

uint shape_at(uint axis) {
    if (axis < 4) return component(shape0, axis);
    return component(shape1, axis - 4);
}

uint stride_at(uint axis) {
    if (axis < 4) return component(strides0, axis);
    return component(strides1, axis - 4);
}

bool is_reduced(uint axis) {
    if (axis < 4) return component(reduced0, axis) != 0;
    return component(reduced1, axis - 4) != 0;
}

uint input_base(uint outputIndex) {
    uint result = header1.y;
    uint remaining = outputIndex;
    for (int axis = int(header0.z) - 1; axis >= 0; --axis) {
        if (!is_reduced(uint(axis))) {
            uint dimension = shape_at(uint(axis));
            uint coordinate = remaining % dimension;
            remaining /= dimension;
            result += coordinate * stride_at(uint(axis));
        }
    }
    return result;
}

uint term_index(uint base, uint term) {
    uint result = base;
    uint remaining = term;
    for (int axis = int(header0.z) - 1; axis >= 0; --axis) {
        if (is_reduced(uint(axis))) {
            uint dimension = shape_at(uint(axis));
            uint coordinate = remaining % dimension;
            remaining /= dimension;
            result += coordinate * stride_at(uint(axis));
        }
    }
    return result;
}

[numthreads(1, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID) {
    uint outputIndex = id.x;
    if (outputIndex >= header0.w) return;
    uint domainCount = header1.x;
    float value = 0.0;
    uint base = input_base(outputIndex);
    for (uint term = 0; term < domainCount; ++term) {
        float next = inputValues[term_index(base, term)];
        if (header0.x <= 1) value += next;
        else if (term == 0 || isnan(next)) value = next;
        else if (!isnan(value) && header0.x == 2 && (next < value || (next == 0.0 && value == 0.0 && (asuint(next) & 0x80000000) != 0))) value = next;
        else if (!isnan(value) && header0.x == 3 && (next > value || (next == 0.0 && value == 0.0 && (asuint(next) & 0x80000000) == 0))) value = next;
    }
    if (header0.x == 1) value /= float(domainCount);
    outputValues[outputIndex] = value;
}
