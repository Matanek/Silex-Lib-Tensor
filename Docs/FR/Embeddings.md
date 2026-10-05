# Apprendre des vecteurs à partir d’indices

`Neural.Embedding(name, vocabulary_size, embedding_features, seed,
padding_index:null)` associe chaque indice `int32` à une ligne d’une table
`float32` entraînable. Une entrée `[tokens]` produit `[tokens, features]` ;
une entrée batch-first `[batch, tokens]` produit `[batch, tokens, features]`.
Les indices répétés additionnent leurs contributions au gradient dense du poids.

```sx
use Tensor
use Tensor.Neural
use Tensor.Optim

func main() {
    var embedding=Neural.Embedding("tokens",4,2,seed:7,padding_index:0)
    var ids:int32[]=[0,1,2,3]
    let input=Tensor.vector(ids)
    var targets:float[]=[0.0,0.0,0.2,-0.3,0.4,-0.5,0.6,-0.7]
    let target=Tensor(targets,[4,2])
    var optimizer=Optim.SGD(embedding.parameters(),learning_rate:0.2)
    var step=0
    while step<40 {
        optimizer.zero_grad()
        let difference=embedding.forward(input).subtract(target)
        difference.multiply(difference).mean().backward()
        optimizer.step()
        step++
    }
    let weights=embedding.parameters()[0].value().detach().values()
    assert(weights[0]==0.0 && weights[1]==0.0)
    assert(embedding.forward(input).shape()[1]==2)
    print("Embedding training: padding preserved")
}
```

L’[exemple exécutable](../../Tests/Consumer/Examples/EmbeddingTraining.sx) utilise
seulement les API publiques. Le poids s’appelle `name + ".weight"` et sa forme
est `[vocabulary_size, embedding_features]`. La seed initialise uniformément la
table dans `[-1, 1]` ; la ligne de padding, si présente, est ensuite mise à zéro.
Le vocabulaire contient entre 1 et 2147483648 lignes et features est positif ;
les contrôles usuels de cardinalité et d’allocation de Tensor s’appliquent.

## Padding et état du modèle

`padding_index` est facultatif et appartient au vocabulaire. Le forward lit sa
ligne normalement, y compris après un remplacement explicite du poids. Son
**gradient de lookup est arrêté**. Ce contrat se vérifie par oracle de sélection,
pas par différences finies sur la ligne arrêtée. Le Jacobien complet sans
padding est qualifié par différences finies.

La ligne initialement nulle reste nulle sous SGD, Adam et AdamW, y compris avec
momentum ou décroissance du poids, tant qu’aucune autre opération ne lui fournit
un gradient et qu’aucune mutation ou restauration ne la remplace. Une décroissance
peut modifier une ligne non nulle remplacée par l’application ; Embedding ne
réinitialise pas les moments d’optimiseur et ne restaure pas une ligne après un pas.

Embedding suit `Layer` : composition `Sequential`, `parameters`, gel/dégel,
modes, transferts explicites et sauvegardes de poids. `TrainingCheckpoint`
conserve sa configuration, y compris le padding, et refuse une configuration
incompatible avant mutation. Les noms de paramètres restent uniques dans un
modèle composé. Une couche gelée retourne un calcul sans provenance de son poids.

## Placement et indices invalides

Les indices et le poids doivent partager placement et device. Utilisez
`embedding.to(device)` avec `input.to(device)`, ou `embedding.cpu()` avec
`input.cpu()` ; la lecture des valeurs GPU impose `detach().cpu().values()`.
Lookup et dérivée restent sur GPU sans upload/download implicite. Les vues
stridées et les axes vides sont acceptés.

Sur CPU, un indice négatif ou hors vocabulaire provoque une erreur avant la
collecte. Sur GPU, le contrat existant de `gather` produit NaN pour ce vecteur
et ignore sa contribution au poids, avec accès mémoire contrôlé. Validez les
indices CPU avant leur transfert si votre application exige un rejet immédiat.
Aucune conversion de dtype, extraction GPU ou normalisation d’indice n’est implicite.
