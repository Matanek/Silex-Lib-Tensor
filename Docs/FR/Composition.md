# Composer une couche dans votre application

Dérivez `Tensor.Neural.Layer` pour donner un contrat commun à votre modèle.
Son constructeur protégé est accessible aux classes dérivées hors du package.
Redéfinissez `forward` avec les opérations Tensor et les couches publiques.

Ce programme compose deux résidus qui partagent la même Dense :

```sx
use Tensor
use Tensor.Autograd
use Tensor.Neural
use GFX.GPU

nocopy class Residual:Neural.Layer {
    private var block:Neural.Dense

    init(block:Neural.Dense) { self.block = block }
    override func forward(input:Tensor) Tensor { return input.add(self.block.forward(input)) }
    override func parameters() Neural.Parameter[] { return self.block.parameters() }
    override func to(device:GPU.Device) { self.block.to(device) }
    override func cpu() { self.block.cpu() }
}

func main() {
    var block = Neural.Dense("shared",1,1,seed:0)
    var parameters = block.parameters()
    parameters[0].replace(Tensor.full([1,1],2.0))
    parameters[1].replace(Tensor.full([1],1.0))
    var left:Neural.Layer = Residual(block)
    var right:Neural.Layer = Residual(block)
    assert(left.parameters()[0] == right.parameters()[0])
    var observations:float[] = [1.0,2.0]
    var input = Autograd.Variable(Tensor(observations,[2,1]))
    let output = left.forward(input.value()).add(right.forward(input.value())).sum()
    assert(output.detach().item() == 22.0)
    var gradients = Autograd.grad(output,[parameters[0].value(),parameters[1].value(),input.value()])
    assert(gradients[0].item() == 6.0 && gradients[1].item() == 4.0)
    for value in gradients[2].values() { assert(value == 6.0) }
    assert(parameters[0].gradient() == null && parameters[1].gradient() == null && input.gradient() == null)
}
```

Le poids vaut 2 et le biais 1. Chaque résidu calcule x + (2x + 1) ; les deux
branches donnent ensemble [8,14] pour [1,2], puis une somme de 22. Le poids
partagé reçoit un gradient de 6, le biais de 4 et chaque entrée de 6. Le partage
conserve l’identité de Parameter : les branches ne créent pas deux poids.

## Relayer les méthodes de la couche

`forward` est le point d’entrée du calcul. `parameters` rend les paramètres
qui doivent être optimisés. Pour une couche contenant des enfants, relayez
également `train`, `eval`, `to(device)` et `cpu()` selon leur état et placement.
Dans cet exemple, Dense n’a pas d’état train/eval ; seuls les paramètres et
transferts sont relayés. Les implémentations de base rendent une liste de
paramètres vide et des hooks d’état sans effet. Layer sert à la dérivation ;
sa construction directe est refusée.

Les méthodes sont distribuées selon la classe réelle, même via une variable
Layer ou Sequential. Le compilateur infère le contrat modifiable de toute la
famille virtuelle lorsqu’un override modifie l’état ; une base avec hook vide
reste suffisante. Une couche avec état utilise donc un récepteur `var`, sans
marqueur artificiel dans la classe de base.

## Garder les gradients et le placement explicites

L’assemblage utilise les opérations Tensor différentiables ordinaires.
`Autograd.grad` permet de dériver à la fois paramètres partagés et entrées,
sans remplir leurs slots. Sur GPU, les calculs restent résidents ; seuls
`to(device)` et `cpu()` demandés effectuent les transferts.

Transmettre une couche existante à plusieurs parents conserve son identité.
Pour l’optimiseur, collectez chaque Parameter partagé une seule fois.
Sequential déduplique désormais les identités partagées et refuse deux objets
différents portant le même nom. Le [registre des couches](Registration.md)
permet d’hériter de cette collecte plutôt que de redéfinir parameters(). Le [guide des buffers et du gel](State.md)
explique les primitives d’état disponibles.
La [recette d’entraînement](Recipes/Training.md) explique la boucle existante,
et le [guide des gradients](Gradients.md) détaille les slots et la consommation
du graphe.
