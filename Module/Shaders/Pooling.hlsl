StructuredBuffer<float> inputValues : register(t0, space0);
RWStructuredBuffer<float> outputValues : register(u0, space1);

cbuffer Metadata : register(b0, space2) {
    uint4 header0;
    uint4 header1;
    uint4 inputShape;
    uint4 inputStrides;
    uint4 outputShape;
};

[numthreads(1, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID) {
    uint outputIndex = id.x;
    if (outputIndex >= header0.x) return;
    uint remaining = outputIndex;
    uint outputX = remaining % outputShape.w;
    remaining /= outputShape.w;
    uint outputY = remaining % outputShape.z;
    remaining /= outputShape.z;
    uint channel = remaining % outputShape.y;
    uint batch = remaining / outputShape.y;
    float value = header1.x == 0 ? asfloat(0xff800000) : 0.0;
    bool hasValue = false;
    for (uint kernelY = 0; kernelY < header0.y; ++kernelY) {
        int inputY = int(outputY * header0.z + kernelY) - int(header0.w);
        for (uint kernelX = 0; kernelX < header0.y; ++kernelX) {
            int inputX = int(outputX * header0.z + kernelX) - int(header0.w);
            if (inputY >= 0 && inputY < int(inputShape.z) && inputX >= 0 && inputX < int(inputShape.w)) {
                uint inputIndex = header1.y + batch * inputStrides.x + channel * inputStrides.y + uint(inputY) * inputStrides.z + uint(inputX) * inputStrides.w;
                float next = inputValues[inputIndex];
                if (header1.x == 1) value += next;
                else if (!hasValue || next > value || isnan(next)) value = next;
                hasValue = true;
            }
        }
    }
    if (header1.x == 1) value /= float(header0.y * header0.y);
    outputValues[outputIndex] = value;
}
