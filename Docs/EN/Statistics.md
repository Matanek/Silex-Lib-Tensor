# Measure dispersion with an explicit correction

`variance` and `standard_deviation` measure dispersion of float32 values on
CPU/GPU. Always specify the statistical correction: zero for the population,
one for the usual sample estimator.

This program checks a variance and its gradient:

```sx
use Tensor
use Tensor.Autograd
use STD.Math

func main() {
    var data:float[] = [1.0,2.0,3.0]
    var input = Autograd.Variable(Tensor.vector(data))
    let population = input.value().variance(correction:0)
    assert(Math.abs(population.detach().item()-2.0/3.0) < 0.001)
    let sample = input.value().variance(correction:1)
    var gradients = Autograd.grad(sample,[input.value()])
    assert(gradients[0].values()[0] == -1.0 && gradients[0].values()[2] == 1.0)
    assert(input.gradient() == null)
    assert(Tensor.vector(data).standard_deviation(correction:1).item() == 1.0)
}
```

The domain contains three values with a mean of two. Their squared deviations
sum to two: population variance 2/3, sample variance one, and sample standard
deviation one. `Autograd.grad` leaves the slot absent.

## Choose the domain and shape

`variance(correction:int, axes:null, keep_dimensions:false)` and
`standard_deviation(correction:int, axes:null, keep_dimensions:false)` use the
same axis conventions as `sum` and `mean`. An absent list reduces every axis;
an empty list treats each value as a singleton domain. Explicit axes are
nonnegative, unique, and within the rank. With `keep_dimensions:true`, each
reduced axis retains size one.

Correction must be a nonnegative integer. Each domain's value count N must be
strictly larger than correction, even when the resulting shape is empty.
An empty domain is therefore rejected; an empty output remains possible when
domains are valid, for example [0,3] reduced on axis1. Scalars and singletons
accept correction0 and return zero for finite values. Integers require an explicit float32 cast before computation. Booleans are
rejected and casting them to numbers remains forbidden.

## Values and gradients

Variance first computes the mean, then centered deviations, and divides their
squared sum by `N - correction`. It avoids the difference-of-moments formula
without promising precision beyond float32 computation. Standard deviation
is its square root. No epsilon, clipping, or implicit correction is added.
NaN and infinities follow float32 operations: a domain containing an infinite or NaN value produces
NaN dispersion here.

The variance gradient is `2 * (x - mean) / (N - correction)`; the standard
deviation gradient is `(x - mean) / ((N - correction) * standard_deviation)`
when standard deviation is positive. When the computed variance is zero, as on an ordinary constant domain, the
standard deviation derivative is not unique; Tensor chooses zero for all inputs.
Perform gradchecks away from this point. Nonfinite values remain observable,
without masking.

Strided views, offsets, expansions, and multiple axes are supported. Gradients
remain first order; `backward` accumulates in slots, `Autograd.grad` preserves
them. GPU float32 computation and gradients stay resident within the existing
rank limit of five, without implicit upload or readback. Observe a tracked
value through `detach().cpu()` and a returned gradient through `cpu()`.

See [batch shapes](Shapes.md), [gradient checking](Gradients.md), and the
[compatibility reference](Reference.md).
