StructuredBuffer<float> inputValues : register(t0, space0);
StructuredBuffer<float> kernelValues : register(t1, space0);
RWStructuredBuffer<float> outputValues : register(u0, space1);

cbuffer Metadata : register(b0, space2) {
    uint4 header0;
    uint4 header1;
    uint4 inputShape;
    uint4 inputStrides;
    uint4 kernelShape;
    uint4 kernelStrides;
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
    uint outputChannel = remaining % outputShape.y;
    uint batch = remaining / outputShape.y;
    float value = 0.0;
    for (uint inputChannel = 0; inputChannel < inputShape.y; ++inputChannel) {
        for (uint kernelY = 0; kernelY < kernelShape.z; ++kernelY) {
            int inputY = int(outputY * header0.y + kernelY) - int(header0.z);
            for (uint kernelX = 0; kernelX < kernelShape.w; ++kernelX) {
                int inputX = int(outputX * header0.y + kernelX) - int(header0.z);
                if (inputY >= 0 && inputY < int(inputShape.z) && inputX >= 0 && inputX < int(inputShape.w)) {
                    uint inputIndex = header0.w + batch * inputStrides.x + inputChannel * inputStrides.y + uint(inputY) * inputStrides.z + uint(inputX) * inputStrides.w;
                    uint kernelIndex = header1.x + outputChannel * kernelStrides.x + inputChannel * kernelStrides.y + kernelY * kernelStrides.z + kernelX * kernelStrides.w;
                    value += inputValues[inputIndex] * kernelValues[kernelIndex];
                }
            }
        }
    }
    outputValues[outputIndex] = value;
}
