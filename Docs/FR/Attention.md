# Appliquer une attention aux valeurs

`Neural.attention(query,key,value,mask:null,scale:null,return_weights:false)`
pondère les values depuis les produits query/key. Elle retourne
`AttentionResult.output` et, sur demande, `AttentionResult.weights`.
Les [masques booléens](SequenceMasks.md) autorisent les positions `true` ;
une query sans clé visible produit une sortie et une contribution de gradient nulles.

Ce programme applique deux queries à trois values, avec la dernière clé exclue.
Il vérifie les moyennes obtenues et les gradients des values sur CPU et Metal.

```sx
use Tensor
use Tensor.Neural
use Tensor.Autograd
use Tensor.Profile
use GFX.GPU

func demonstrate_attention(device:GPU.Device?) {
    var samples:float[]=[2.0,4.0,6.0,8.0,10.0,12.0]
    var q=Tensor.zeros([1,2,2]);var k=Tensor.zeros([1,3,2]);var v=Tensor(samples,[1,3,2])
    var mask=Neural.sequence_mask([2],3).reshape([1,1,3])
    if let gpu=device {q=q.to(gpu);k=k.to(gpu);v=v.to(gpu);mask=mask.to(gpu)}
    var values=Autograd.Variable(v)
    var profile:Profile?=null
    if let gpu=device {profile=Profile(gpu)}
    let result=Neural.attention(q,k,values.value(),mask:mask,return_weights:true)
    result.output.sum().backward()
    if let p=profile {assert(p.snapshot().uploads==0 && p.snapshot().downloads==0)}
    let output=result.output.detach().cpu().values()
    assert(output[0]==4.0 && output[1]==6.0 && output[2]==4.0 && output[3]==6.0)
    if let weights=result.weights {assert(weights.detach().cpu().values()[2]==0.0)}
    else {panic("expected attention weights")}
    if let gradient=values.gradient() {
        let data=gradient.cpu().values()
        assert(data[0]==1.0 && data[1]==1.0 && data[2]==1.0 && data[3]==1.0 && data[4]==0.0 && data[5]==0.0)
    } else {panic("expected value gradient")}
}
func main() {
    demonstrate_attention(null)
    if GPU.Device.is_supported() {var device=GPU.Device(GPU.DeviceSettings(debug:true));demonstrate_attention(device)}
    print("Attention: visible values averaged and gradients preserved")
}
```

Les deux sorties valent `[4,6]`. Chaque value visible reçoit un gradient `[1,1]`
car elle contribue pour moitié aux deux sorties ; la value exclue reçoit `[0,0]`.
Les uploads précèdent le profil ; le calcul et le backward restent résidents.

## Choisir les formes et la visibilité

Q, K et V sont `float32`, de même rang3 ou4, placement et device. Les formes
batch-first sont `[batch,queries,features]`, `[batch,keys,features]` et
`[batch,keys,value_features]`. Avec des têtes, ajoutez un axe après batch :
`[batch,heads,sequence,features]`. Batch et heads correspondent exactement,
sans broadcasting entre Q/K/V ; Q/K features sont identiques et strictement
positives. K et V ont le même nombre de keys. Le résultat conserve les axes
batch, heads et queries, avec les features de V.

Le masque booléen se broadcaste vers `[batch,queries,keys]` ou
`[batch,heads,queries,keys]` sans agrandir ces formes. `null` autorise toutes les
clés sur le placement des entrées. Composez explicitement longueur et causalité
avec `logical_and` ; pour plusieurs têtes, un masque de padding `[batch,keys]`
prend la forme `[batch,1,1,keys]`. Transférez le masque avant l'appel GPU.
Les poids optionnels ont la forme des scores et restent différentiables.

La self-attention emploie la même source pour Q/K/V ; la cross-attention accepte
des longueurs queries et keys distinctes. Le partage d'une variable accumule les
contributions des trois chemins. `Autograd.grad` accepte une seed non uniforme ;
un graphe consommé ne peut pas être réutilisé pour un second backward.

## Échelle, stabilité et formes vides

L'échelle par défaut est `1/sqrt(features)` ; `scale` la remplace par un scalaire
fini, qui peut être zéro ou négatif. Les valeurs et les scores visibles issus
du produit et de l'échelle doivent rester finis en float32. Le
[softmax masqué stable](MaskedSoftmax.md) soustrait un maximum avant exponentiation
et définit les lignes entièrement exclues à zéro. Les arrondis restent float32.

Batch, heads, queries, keys et value_features peuvent être nuls. Les formes de
sortie et de gradient sont conservées ; zéro key donne une sortie nulle même
lorsqu'il existe des queries. Les vues stridées sont acceptées. Il n'y a aucune
projection entraînable ou dropout dans cette fonction : l'application choisit
ses projections et assemble les opérations avec les [couches composables](Composition.md).
