StructuredBuffer<float> leftValues : register(t0, space0);
StructuredBuffer<float> rightValues : register(t1, space0);
RWStructuredBuffer<float> outputValues : register(u0, space1);

cbuffer Metadata : register(b0, space2) {
    uint4 header0;
    uint4 header1;
    uint4 shape0;
    uint4 shape1;
    uint4 leftStrides0;
    uint4 leftStrides1;
    uint4 rightStrides0;
    uint4 rightStrides1;
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

uint stride_at(uint4 first, uint4 second, uint axis) {
    if (axis < 4) return component(first, axis);
    return second.x;
}

uint storage_index(uint logical, uint offset, uint4 first, uint4 second) {
    uint result = offset;
    uint remaining = logical;
    for (int axis = int(header0.z) - 1; axis >= 0; --axis) {
        uint dimension = shape_at(uint(axis));
        uint coordinate = dimension == 0 ? 0 : remaining % dimension;
        if (dimension > 0) remaining /= dimension;
        result += coordinate * stride_at(first, second, uint(axis));
    }
    return result;
}

[numthreads(1, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID) {
    uint index = id.x;
    if (index >= header0.y) return;
    float left = leftValues[storage_index(index, header0.w, leftStrides0, leftStrides1)];
    float right = rightValues[storage_index(index, header1.x, rightStrides0, rightStrides1)];
    if (header0.x == 0) outputValues[index] = left + right;
    else if (header0.x == 1) outputValues[index] = left - right;
    else if (header0.x == 2) outputValues[index] = left * right;
    else if (header0.x == 3) outputValues[index] = left / right;
    else outputValues[index] = left == right ? 1.0 : 0.0;
}
