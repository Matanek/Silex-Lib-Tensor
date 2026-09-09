# Entraîner un MLP, un CNN ou un RNN

Ces recettes changent l'architecture et la forme des données, mais conservent
la même boucle eager. Les sources complètes appartiennent au consommateur public
de Tensor et peuvent être lancées depuis la racine du workspace.

## Boucle commune

Une étape d'entraînement remet d'abord les gradients à zéro, exécute le modèle,
construit une perte scalaire, consomme son graphe puis remplace les valeurs des
paramètres :

```sx
optimizer.zero_grad()
let loss = model.forward(input).cross_entropy(target, 1)
loss.backward()
optimizer.step()
```

`Tensor` reste une valeur immuable. `NN.Parameter` possède la feuille
entraînable ; `optimizer.step()` lui affecte une nouvelle valeur détachée.
Lire `loss.item()` dans chaque itération est correct sur CPU, mais synchronise
un modèle GPU. Espacez cette observation lorsque la boucle doit rester
résidente.

## MLP dense

Le [MLP exécutable](../../../Tests/Consumer/Examples/TrainMLP.sx) apprend un
problème XOR, enregistre ses paramètres puis recharge un modèle neuf. Son
architecture est :

```sx
var layers:NN.Layer[] = [
    NN.Dense("hidden", 2, 8, 101),
    NN.Activation.tanh(),
    NN.Dense("classifier", 8, 2, 102)
]
var model = NN.Sequential(layers)
```

Les noms de couches déterminent les noms stables du checkpoint. Réutiliser un
nom dans la même collection est refusé avant l'entraînement.

## CNN NCHW

Le [CNN exécutable](../../../Tests/Consumer/Examples/TrainCNN.sx) classe quatre
motifs `4 × 4`. Tensor expose NCHW pour les entrées et OIHW pour les noyaux :

```sx
var layers:NN.Layer[] = [
    NN.Conv2D("features", 1, 4, 2, 211),
    NN.Activation.relu(),
    NN.MaxPool2D(2, stride:1),
    NN.Flatten(),
    NN.Dense("classifier", 16, 2, 212)
]
```

Le `Flatten` préserve la dimension de batch. La taille d'entrée de `Dense`
doit correspondre exactement aux canaux et dimensions spatiales restants.

## RNN simple

Le [RNN exécutable](../../../Tests/Consumer/Examples/TrainRNN.sx) reçoit
`[batch, temps, features]`. `SimpleRNN` déroule une récurrence tanh
unidirectionnelle et rend uniquement l'état final :

```sx
var layers:NN.Layer[] = [
    NN.SimpleRNN("memory", 1, 6, 307),
    NN.Dense("classifier", 6, 2, 308)
]
```

La recette écrête la norme globale avant `step()`. Cette opération compose tous
les gradients présents et ne crée pas une limite indépendante par paramètre.

## Garder l'étape sur GPU

Construisez d'abord le modèle et les données, placez-les sur le même device,
puis créez l'optimiseur depuis les paramètres déjà placés :

```sx
var device = GPU.Device()
model.to(device)
let gpu_input = input.to(device)
let gpu_target = target.to(device)
var parameters = model.parameters()
var optimizer = Optim.Adam(parameters, learning_rate:0.01)

optimizer.zero_grad()
let loss = model.forward(gpu_input).cross_entropy(gpu_target, 1)
loss.backward()
optimizer.step()

let observed = loss.detach().cpu().item()
```

Forward, backward et `step()` restent résidents. Le dernier `cpu()` est le
readback et le point d'attente volontaire. Les trois parcours CPU/GPU complets
sont vérifiés par les tests `TrainingMLPGPU.sx`, `TrainingCNNGPU.sx` et
`TrainingRNNGPU.sx`.

## Sauvegarder les paramètres

```sx
NN.Checkpoint.save(model, path)
var restored = make_model(999)
NN.Checkpoint.load(restored, path)
```

Chargez sur CPU un modèle de même architecture et de mêmes noms, puis appelez
`restored.to(device)` si nécessaire. Le checkpoint ne reprend pas l'état Adam ;
une reprise exacte de l'optimiseur doit être gérée séparément par l'application.
