StructuredBuffer<float> gradientValues : register(t0, space0);
StructuredBuffer<float> otherValues : register(t1, space0);
RWStructuredBuffer<float> outputValues : register(u0, space1);

cbuffer Metadata : register(b0, space2) {
    uint4 header0;
    uint4 header1;
    uint4 gradientShape;
    uint4 gradientStrides;
    uint4 otherShape;
    uint4 otherStrides;
    uint4 resultShape;
};

uint gradient_index(uint batch, uint channel, uint y, uint x) {
    return header1.x + batch * gradientStrides.x + channel * gradientStrides.y + y * gradientStrides.z + x * gradientStrides.w;
}

uint other_index(uint first, uint second, uint third, uint fourth) {
    return header1.y + first * otherStrides.x + second * otherStrides.y + third * otherStrides.z + fourth * otherStrides.w;
}

[numthreads(1, 1, 1)]
void compute_main(uint3 id : SV_DispatchThreadID) {
    uint logical = id.x;
    if (logical >= header0.x) return;
    uint remaining = logical;
    uint fourth = remaining % resultShape.w;
    remaining /= resultShape.w;
    uint third = remaining % resultShape.z;
    remaining /= resultShape.z;
    uint second = remaining % resultShape.y;
    uint first = remaining / resultShape.y;
    float value = 0.0;

    if (header0.y == 0) {
        uint batch = first;
        uint inputChannel = second;
        uint inputY = third;
        uint inputX = fourth;
        for (uint outputChannel = 0; outputChannel < otherShape.x; ++outputChannel) {
            for (uint kernelY = 0; kernelY < otherShape.z; ++kernelY) {
                int outputYNumerator = int(inputY) + int(header0.w) - int(kernelY);
                for (uint kernelX = 0; kernelX < otherShape.w; ++kernelX) {
                    int outputXNumerator = int(inputX) + int(header0.w) - int(kernelX);
                    if (outputYNumerator >= 0 && outputXNumerator >= 0 &&
                        outputYNumerator % int(header0.z) == 0 && outputXNumerator % int(header0.z) == 0) {
                        uint outputY = uint(outputYNumerator) / header0.z;
                        uint outputX = uint(outputXNumerator) / header0.z;
                        if (outputY < gradientShape.z && outputX < gradientShape.w) {
                            value += gradientValues[gradient_index(batch, outputChannel, outputY, outputX)] *
                                otherValues[other_index(outputChannel, inputChannel, kernelY, kernelX)];
                        }
                    }
                }
            }
        }
    } else {
        uint outputChannel = first;
        uint inputChannel = second;
        uint kernelY = third;
        uint kernelX = fourth;
        for (uint batch = 0; batch < otherShape.x; ++batch) {
            for (uint outputY = 0; outputY < gradientShape.z; ++outputY) {
                int inputY = int(outputY * header0.z + kernelY) - int(header0.w);
                for (uint outputX = 0; outputX < gradientShape.w; ++outputX) {
                    int inputX = int(outputX * header0.z + kernelX) - int(header0.w);
                    if (inputY >= 0 && inputY < int(otherShape.z) && inputX >= 0 && inputX < int(otherShape.w)) {
                        value += gradientValues[gradient_index(batch, outputChannel, outputY, outputX)] *
                            otherValues[other_index(batch, inputChannel, uint(inputY), uint(inputX))];
                    }
                }
            }
        }
    }
    outputValues[logical] = value;
}
