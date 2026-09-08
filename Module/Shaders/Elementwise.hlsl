StructuredBuffer<float> inputValues : register(t0, space0);
RWStructuredBuffer<float> outputValues : register(u0, space1);

cbuffer Metadata : register(b0, space2) {
    uint4 header;
    uint4 shape0;
    uint4 shape1;
    uint4 strides0;
    uint4 strides1;
};

cbuffer ScalarUniform : register(b1, space2) {
    float scalar;
};

uint component(uint4 value, uint index) {
    if (index == 0) return value.x;
    if (index == 1) return value.y;
    if (index == 2) return value.z;
    return value.w;
}

uint shape_at(uint axis) {
    if (axis < 4) return component(shape0, axis);
    return shape1.x;
}

uint stride_at(uint axis) {
    if (axis < 4) return component(strides0, axis);
    return strides1.x;
}

uint storage_index(uint logical) {
    uint result = header.w;
    uint remaining = logical;
    for (int axis = int(header.z) - 1; axis >= 0; --axis) {
        uint dimension = shape_at(uint(axis));
        uint coordinate = dimension == 0 ? 0 : remaining % dimension;
        if (dimension > 0) remaining /= dimension;
        result += coordinate * stride_at(uint(axis));
    }
    return result;
}

[numthreads(1, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID) {
    uint index = id.x;
    if (index >= header.y) return;
    if (header.x == 14) {
        outputValues[index] = scalar;
        return;
    }
    float value = inputValues[storage_index(index)];
    if (header.x == 0) outputValues[index] = value + scalar;
    else if (header.x == 1) outputValues[index] = value - scalar;
    else if (header.x == 2) outputValues[index] = value * scalar;
    else if (header.x == 3) outputValues[index] = value / scalar;
    else if (header.x == 4) outputValues[index] = -value;
    else if (header.x == 5) outputValues[index] = abs(value);
    else if (header.x == 6) outputValues[index] = exp(value);
    else if (header.x == 7) outputValues[index] = log(value);
    else if (header.x == 8) outputValues[index] = sqrt(value);
    else if (header.x == 9) outputValues[index] = isnan(value) ? value : max(value, 0.0);
    else if (header.x == 10) outputValues[index] = 1.0 / (1.0 + exp(-value));
    else if (header.x == 11) outputValues[index] = tanh(value);
    else if (header.x == 12) outputValues[index] = value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0);
    else outputValues[index] = value > 0.0 ? 1.0 : 0.0;
}
