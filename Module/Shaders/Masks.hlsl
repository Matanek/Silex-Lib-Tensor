ByteAddressBuffer leftValues : register(t0, space0);
ByteAddressBuffer rightValues : register(t1, space0);
ByteAddressBuffer thirdValues : register(t2, space0);
RWByteAddressBuffer outputValues : register(u0, space1);

cbuffer Metadata : register(b0, space2) {
    uint4 header0;
    uint4 header1;
    uint4 shape0;
    uint4 shape1;
    uint4 leftStrides0;
    uint4 leftStrides1;
    uint4 rightStrides0;
    uint4 rightStrides1;
    uint4 thirdStrides0;
    uint4 thirdStrides1;
};
cbuffer Scalar : register(b1, space2) { float scalar; };

uint component(uint4 value, uint index) {
    if (index == 0) return value.x;
    if (index == 1) return value.y;
    if (index == 2) return value.z;
    return value.w;
}
uint storage_index(uint logical, uint offset, uint4 first, uint4 second) {
    uint result = offset;
    for (int axis = int(header0.z) - 1; axis >= 0; --axis) {
        uint dimension = axis < 4 ? component(shape0, uint(axis)) : shape1.x;
        uint coordinate = dimension == 0 ? 0 : logical % dimension;
        if (dimension > 0) logical /= dimension;
        uint stride = axis < 4 ? component(first, uint(axis)) : second.x;
        result += coordinate * stride;
    }
    return result;
}
bool left_flag(uint index) {
    return ((leftValues.Load((index / 4) * 4) >> ((index % 4) * 8)) & 255) != 0;
}
bool right_flag(uint index) {
    return ((rightValues.Load((index / 4) * 4) >> ((index % 4) * 8)) & 255) != 0;
}
bool compare_at(uint logical) {
    uint a = storage_index(logical, header0.w, leftStrides0, leftStrides1);
    uint b = storage_index(logical, header1.x, rightStrides0, rightStrides1);
    if (header1.z != 0) {
        bool left = left_flag(a);
        bool right = header1.w != 0 ? scalar != 0.0 : right_flag(b);
        if (header0.x == 0) return left == right;
        if (header0.x == 1) return left != right;
        if (header0.x == 6) return !left;
        if (header0.x == 7) return left && right;
        return left || right;
    }
    float left = asfloat(leftValues.Load(a * 4));
    float right = header1.w != 0 ? scalar : asfloat(rightValues.Load(b * 4));
    if (header0.x == 0) return left == right;
    if (header0.x == 1) return left != right;
    if (header0.x == 2) return left < right;
    if (header0.x == 3) return left <= right;
    if (header0.x == 4) return left > right;
    return left >= right;
}
[numthreads(1, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID) {
    if (header0.x == 9) {
        uint logical = id.x;
        if (logical >= header0.y) return;
        uint m = storage_index(logical, header0.w, leftStrides0, leftStrides1);
        uint a = storage_index(logical, header1.x, rightStrides0, rightStrides1);
        uint b = storage_index(logical, header1.y, thirdStrides0, thirdStrides1);
        uint value = left_flag(m) ? rightValues.Load(a * 4) : thirdValues.Load(b * 4);
        outputValues.Store(logical * 4, value);
        return;
    }
    // One writer owns each packed word, including zeroed trailing padding.
    uint base = id.x * 4;
    if (base >= header0.y) return;
    uint packed = 0;
    for (uint byte = 0; byte < 4 && base + byte < header0.y; ++byte) {
        packed |= (compare_at(base + byte) ? 1u : 0u) << (byte * 8);
    }
    outputValues.Store(id.x * 4, packed);
}
