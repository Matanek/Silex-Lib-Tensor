# Garder une métadonnée du modèle sur CPU

Un compteur exact ou un indicateur de contrôle peut rester sur CPU pendant que
les paramètres et statistiques numériques passent sur GPU. Déclarez ce choix
avec `BufferPlacement.cpu`. Le parcours `Layer.to(device)` conserve alors le
buffer CPU et transfère le reste de l’état enregistré.

Cet exemple garde son compteur int64 sur CPU et applique son paramètre sur le
placement de l’entrée. L’observation finale du résultat GPU est explicite.

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

`Buffer(name,value,placement:BufferPlacement.model)` reste le défaut : ce buffer
suit les déplacements du modèle, avec le comportement déjà établi.
`BufferPlacement.cpu` exige une valeur initiale CPU, sans téléchargement caché
ni conversion. `placement_policy()` retourne le choix immuable. `replace`
préserve encore dtype, forme et placement ; les valeurs capturées et les
références partagées conservent leurs contrats.

`model.to(device)` laisse les buffers CPU déclarés sur CPU ; `model.cpu()` les
y laisse aussi. Un `buffer.to(device)` direct sur un tel buffer échoue avant
mutation, puisqu’il contredit sa politique. Les autres buffers et paramètres
restent prévalidés et transférés une fois par identité enregistrée unique.
Modes, gel, noms et collecte restent inchangés.

Cette politique décrit une métadonnée hôte. Elle ne convertit pas un calcul GPU
en calcul CPU : une opération Tensor entre CPU et GPU reste incompatible. Une
valeur CPU d’état ne devient pas une donnée GPU implicite ; déclarez les buffers
numériques avec la politique `model` lorsqu’ils doivent participer au calcul.
