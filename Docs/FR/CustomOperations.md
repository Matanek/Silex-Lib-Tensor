# Ajouter une opération différentiable

Implémentez `Autograd.Operation`, puis appelez `Autograd.apply(operation, inputs)`
pour relier votre calcul aux feuilles suivies. `forward` et `backward` utilisent
seulement l'API publique de Tensor ; `gradcheck` contrôle la règle sur CPU.

Ce programme définit un carré élément par élément et vérifie son gradient :

```sx
use Tensor
use Tensor.Autograd

struct Square:Autograd.Operation {
    func forward(inputs:Tensor[]) Tensor {
        return inputs[0].multiply(inputs[0])
    }
    func backward(inputs:Tensor[], output:Tensor, gradient:Tensor) Tensor[] {
        return [inputs[0].multiply(2.0).multiply(gradient)]
    }
}
func energy(inputs:Tensor[]) Tensor { return Autograd.apply(Square(), inputs).sum() }
func main() {
    var values:float[] = [2.0, 3.0]
    var input = Autograd.Variable(Tensor.vector(values))
    energy([input.value()]).backward()
    if let gradient = input.gradient() {
        assert(gradient.values()[0] == 4.0 && gradient.values()[1] == 6.0)
    } else { panic("missing gradient") }
    let check = Autograd.gradcheck(energy, [Tensor.vector(values)],
        epsilon:0.001, absolute_tolerance:0.003, relative_tolerance:0.002)
    assert(check.passed && check.checked_derivatives == 2)
    print("Custom gradient: 4, 6; gradcheck: pass")
}
```

La sortie est `Custom gradient: 4, 6; gradcheck: pass`.
L'[exemple complet](../../Tests/Consumer/Examples/CustomOperation.sx) est un
consommateur extérieur au package.

## Définir la règle

`forward(inputs:Tensor[]) Tensor` reçoit les valeurs détachées des entrées et
retourne un Tensor détaché. `backward(inputs:Tensor[], output:Tensor,
gradient:Tensor) Tensor[]` reçoit ces mêmes valeurs, la sortie détachée du
forward et le gradient amont détaché. Il retourne **une contribution par entrée**,
dans le même ordre, y compris pour les entrées non suivies.

Chaque contribution a exactement la forme, le placement et le device de son
entrée, et utilise `float32`. Le Tensor retourné par forward peut changer de
forme ; ses entrées sont non vides et toutes en `float32`, sur le même placement
et device. Aucune conversion, copie CPU/GPU ou adaptation du gradient n'est
implicite. Une règle de broadcast doit donc réduire explicitement ses contributions
aux formes d'entrée. Les vues et formes vides suivent les opérations publiques.

Répéter une entrée dans apply additionne ses contributions ; une entrée non
suivie participe au calcul sans recevoir de slot. Sans entrée suivie, apply
retourne directement la sortie détachée et ne retient pas de règle de graphe.
Les seeds de backward et grad gardent leurs conventions habituelles. Les
contributions sont détachées : seules les dérivées de premier ordre sont prises
en charge. Un Tensor ou un graphe déjà consommé reste refusé.

## Préserver le graphe et les accumulateurs

Les entrées détachées, la sortie et la valeur d'opération restent possédées par
le graphe jusqu'à sa consommation réussie ou sa destruction. Après consommation,
ces références sont libérées même si la sortie calculée reste vivante. Les
références gardées explicitement par l'application ont leur propre durée de vie.

Les additions de gradients de feuilles sont préparées avant publication des
slots. Lorsqu’un graphe contient une règle personnalisée, toutes ses règles
sont aussi validées avant consommation des opérations. Les graphes composés
uniquement de primitives conservent leur libération progressive des valeurs. Une cardinalité, forme,
dtype, provenance ou placement invalide est refusé avec un diagnostic, sans
accumulation partielle par le moteur. `Autograd.grad` ne publie aucun slot.
Les erreurs suivent le comportement fatal des opérations Tensor ; aucun résultat
récupérable nouveau n'est ajouté.

La règle doit dépendre de ses données d'entrée et ne pas modifier des Variable,
Parameter ou états externes pendant la dérivation. Le moteur ne restaure pas les
effets arbitraires d'un callback, ni les mutations de ses captures. Pour gradcheck,
la règle et le forward doivent en plus être déterministes entre les sondes.

Consultez le [contrôle des gradients](Gradients.md) pour les seeds, le Jacobien
complet et les tolérances, et la [référence](Reference.md) pour les placements.
