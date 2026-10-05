# Entraîner une attention multi-têtes

`Neural.MultiHeadAttention` apprend quatre projections nommées query, key, value
et output. `attend(query,key,value,...)` retourne la sortie et des poids optionnels ;
`forward(input)` fournit une self-attention sans masque, à itération 0. La couche
dérive de [Layer](Composition.md) : enfants, paramètres, mode, placement et gel
suivent le [registre partagé](Registration.md).

Ce programme compose padding des queries et keys avec causalité, puis réalise
un pas AdamW sur CPU et Metal. La troisième position est exclue des deux côtés :
sa sortie et son gradient restent nuls, y compris après la projection finale.

```sx
use Tensor
use Tensor.Neural
use Tensor.Autograd
use Tensor.Optim
use Tensor.Profile
use GFX.GPU

func demonstrate_multihead(device:GPU.Device?) {
    var model=Neural.MultiHeadAttention("attention",4,2,seed:17,dropout_probability:0.1)
    var data:float[]=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
    var input=Tensor(data,[1,3,4])
    let valid=Neural.sequence_mask([2],3)
    var mask=valid.reshape([1,1,3,1]).logical_and(valid.reshape([1,1,1,3])).logical_and(Neural.causal_mask(3,3))
    if let gpu=device {model.to(gpu);input=input.to(gpu);mask=mask.to(gpu)}
    var values=Autograd.Variable(input)
    var optimizer=Optim.AdamW(model.parameters(),learning_rate:0.01)
    var profile:Profile?=null
    if let gpu=device {profile=Profile(gpu)}
    let result=model.attend(values.value(),values.value(),values.value(),mask:mask,iteration:0,return_weights:true)
    result.output.sum().backward();optimizer.step();optimizer.zero_grad()
    if let p=profile {assert(p.snapshot().uploads==0 && p.snapshot().downloads==0)}
    let output=result.output.detach().cpu().values()
    assert(result.output.shape()[2]==4)
    var i=8;while i<12 {assert(output[i]==0.0);i++}
    if let weights=result.weights {assert(weights.shape()[1]==2 && weights.shape()[2]==3 && weights.shape()[3]==3)} else {panic("expected attention weights")}
    if let gradient=values.gradient() {
        let gradient_values=gradient.cpu().values();var j=8;while j<12 {assert(gradient_values[j]==0.0);j++}
    } else {panic("expected input gradient")}
}
func main() {
    demonstrate_multihead(null)
    if GPU.Device.is_supported() {var device=GPU.Device(GPU.DeviceSettings(debug:true));demonstrate_multihead(device)}
    print("Multi-head attention: causal padding and resident AdamW step")
}
```

Définissez les deux masques de longueur lorsque les queries remplies de padding
doivent aussi être exclues. Un masque de keys seul autorise encore ces queries
à lire les keys visibles. Les transferts précèdent le calcul ; le forward,
backward et pas d'optimiseur n'ajoutent aucun upload ou readback.

## Projeter et réunir les têtes

`MultiHeadAttention(name,features,heads,seed,dropout_probability:0.0)` exige
features et heads positifs, avec features divisible par heads. Q/K/V ont forme
`[batch,sequence,features]`, dtype float32, même batch, largeur, placement et
device. Les queries peuvent avoir une longueur différente des keys ; K et V
ont la même longueur. Les quatre poids ont forme `[features,features]` dans
l'ordre `[input,output]`, leurs biais `[features]`.

Les noms de paramètres sont `name.query.weight`, `name.query.bias`, puis les
paires key, value et output dans cet ordre. Les enfants portent les noms query,
key, value et output. Chaque projection place les groupes de features contigus
dans `[batch,heads,sequence,features/heads]`. Leur réunion réordonne les axes et
les matérialise sur le même placement avant la projection output.

La sortie a forme `[batch,queries,features]`. Les poids optionnels gardent toutes
les têtes : `[batch,heads,queries,keys]`. Masque et échelle suivent le
[calcul d'attention](Attention.md), avec échelle par défaut
`1/sqrt(features/heads)`. Une query sans clé visible dans aucune tête conserve
sortie et contribution de gradient nulles, même si le biais output est non nul.
Les dimensions batch, queries ou keys peuvent être nulles. Valeurs et scores
visibles doivent rester finis en float32.

## Contrôler le dropout et reprendre

Le dropout s'applique aux probabilités en mode train. La seed de construction et
l'argument `iteration` de `attend`, tous deux dans `[0,4294967295]`, choisissent
explicitement le tirage. Un appel répété à la même itération reproduit le tirage
sur son backend. En évaluation, le dropout est désactivé. Les poids retournés
sont ceux appliqués aux values, après dropout ; leur somme peut alors différer
de 1. Une ligne rendue vide par le dropout conserve le biais output, tandis
qu'une query sans clé visible reste à zéro.

Utilisez `attend` avec `iteration:progress.step` pour faire progresser les tirages
pendant l'entraînement. `forward(input)` conserve le raccourci Layer à itération 0.
Le [checkpoint complet](TrainingCheckpoint.md) restaure poids, moments, modes,
seed et curseur ; TrainingProgress fournit l'itération du prochain appel. La
reprise sur GPU transfère explicitement poids et moments comme dans ce guide.
Un nombre de têtes ou une probabilité incompatible est refusé avant restauration.
Le checkpoint de paramètres seul conserve son rôle distinct.

Une couche partagée garde une seule collection de huit paramètres et une seule
mise à jour. Gelez le modèle après `zero_grad` ; les gradients des entrées restent
possibles. Cette couche exprime self/cross-attention ; l'application compose
séparément résidus, normalisation, MLP et encodage positionnel.
