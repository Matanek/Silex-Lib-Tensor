# Keep model metadata on CPU

An exact counter or control flag can stay on CPU while parameters and numeric
statistics move to GPU. Declare that choice with `BufferPlacement.cpu`.
`Layer.to(device)` then retains the CPU buffer and transfers the rest of the
registered state.

This example keeps its int64 counter on CPU and applies its parameter on the
input placement. The final observation of the GPU result is explicit.

```sx
use Tensor
use Tensor.Neural
use GFX.GPU

nocopy class CounterLayer:Neural.Layer {
    private var weight:Neural.Parameter
    private var count:Neural.Buffer
    init() {
        self.weight=Neural.Parameter("weight",Tensor.scalar(2.0))
        self.count=Neural.Buffer("count",Tensor.scalar(0 as int64),placement:Neural.BufferPlacement.cpu)
        self.register_parameter(self.weight);self.register_buffer(self.count)
    }
    override func forward(input:Tensor) Tensor {
        self.count.replace(Tensor.scalar(self.count.value().int64_item()+1))
        return input.multiply(self.weight.value())
    }
}

func main() {
    var layer=CounterLayer();var input=Tensor.scalar(3.0)
    if GPU.Device.is_supported() {
        var device=GPU.Device();layer.to(device);input=input.to(device)
    }
    let result=layer.forward(input)
    assert(layer.buffers()[0].value().is_cpu())
    assert(layer.buffers()[0].value().int64_item()==1 && result.detach().cpu().item()==6.0)
}
```

`Buffer(name,value,placement:BufferPlacement.model)` remains the default: the
buffer follows model transfers with its established behavior.
`BufferPlacement.cpu` requires an initial CPU value, with no hidden download or
conversion. `placement_policy()` returns the immutable choice. `replace` still
preserves dtype, shape, and placement; captured values and shared references
retain their contracts.

`model.to(device)` leaves declared CPU buffers on CPU; `model.cpu()` also leaves
them there. Calling `buffer.to(device)` directly on one fails before mutation,
because it conflicts with its policy. Other buffers and parameters are still
prevalidated and transferred once per unique registered identity. Modes,
freezing, names, and collection stay unchanged.

This policy describes host metadata. It does not turn GPU computation into CPU
computation: a Tensor operation between CPU and GPU remains incompatible.
A CPU state value does not implicitly become GPU data; declare numeric buffers
with the `model` policy when they need to participate in computation.
