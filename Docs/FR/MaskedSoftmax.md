# Normaliser les scores visibles

`Tensor.masked_softmax(mask,axis)` normalise uniquement les positions autorisées
par un masque booléen. Une ligne sans position visible retourne des zéros avec
gradients nuls. Le masque suit la convention `true` des
[masques de séquence](SequenceMasks.md).

Ce programme construit les probabilités de trois séquences de longueurs 0, 2 et
4. Il contrôle aussi les gradients et exécute le même parcours sur CPU et Metal
lorsque le GPU est disponible.

```sx
use Tensor
use Tensor.Neural
use Tensor.Autograd
use Tensor.Profile
use GFX.GPU

func demonstrate_masked_probabilities(device:GPU.Device?) {
    var scores=Tensor.ones([3,4])
    var mask=Neural.sequence_mask([0,2,4],4)
    if let gpu=device {scores=scores.to(gpu);mask=mask.to(gpu)}
    var input=Autograd.Variable(scores)
    var profile:Profile?=null
    if let gpu=device {profile=Profile(gpu)}
    let probabilities=input.value().masked_softmax(mask,axis:1)
    probabilities.sum().backward()
    if let p=profile {assert(p.snapshot().uploads==0 && p.snapshot().downloads==0)}
    let values=probabilities.detach().cpu().values()
    assert(values[0]==0.0 && values[1]==0.0 && values[2]==0.0 && values[3]==0.0)
    assert(values[4]==0.5 && values[5]==0.5 && values[6]==0.0 && values[7]==0.0)
    assert(values[8]==0.25 && values[9]==0.25 && values[10]==0.25 && values[11]==0.25)
    if let gradient=input.gradient() {for value in gradient.cpu().values() {assert(value==0.0)}}
    else {panic("expected masked softmax gradient")}
}
func main() {
    demonstrate_masked_probabilities(null)
    if GPU.Device.is_supported() {var device=GPU.Device(GPU.DeviceSettings(debug:true));demonstrate_masked_probabilities(device)}
    print("Masked probabilities: empty row and gradients preserved")
}
```

Les lignes donnent respectivement `[0,0,0,0]`, `[0.5,0.5,0,0]` et
`[0.25,0.25,0.25,0.25]`. La somme des probabilités d’une ligne visible vaut 1 ;
son gradient est nul lorsque la loss est cette somme. Le masque lui-même est
une donnée détachée, jamais un paramètre différentiable.

## Formes et stabilité

Les scores doivent être `float32`. L’axe doit appartenir au rang de l’entrée.
Le masque booléen doit pouvoir être broadcasté vers la forme des scores sans
l’agrandir ; résultat et gradients conservent forme et placement. Les vues stridées
et les axes autres que le dernier sont pris en charge. Une entrée vide retourne
une entrée vide de même forme avec gradient vide, y compris un axe de longueur0.

Le domaine numérique qualifié contient des scores visibles finis, y compris de
grandes translations et les extrêmes finis float32. Les positions exclues peuvent
contenir NaN ou infini : elles donnent une probabilité et une dérivée nulles sans
contaminer les positions visibles. Les valeurs visibles non finies restent hors
de ce domaine ; aucun readback n’est ajouté pour les examiner sur GPU.

Une ligne entière `false` donne des zéros, jamais une distribution uniforme.
Sur une ligne visible, une seed non uniforme donne le gradient habituel de la
normalisation, avec zéros aux positions exclues. `Autograd.grad` préserve les
slots existants ; `backward` suit les règles ordinaires de consommation du graphe.

Le calcul conserve les arrondis float32 ; normalisation et gradients se
comparent avec une tolérance numérique.

## Placer et composer

Scores et masque doivent avoir le même placement et le même device. Transférez
explicitement le masque avant le calcul ; softmax et backward restent résidents
sans upload ou readback implicite. Ajoutez les axes batch/tête avec `unsqueeze`
comme dans le [guide de visibilité](SequenceMasks.md).

Cette opération normalise des scores. Le produit query/key, l’échelle et
l’application des probabilités aux values restent des calculs Tensor distincts.
Les constantes et ressources nécessaires sont libérées avec le graphe selon le
parcours habituel ; aucune nouvelle couche à état n’est créée.
