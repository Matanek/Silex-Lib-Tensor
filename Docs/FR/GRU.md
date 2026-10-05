# Entraîner une GRU avec état explicite

`Neural.GRU(name,input_features,hidden_features,seed)` déroule une couche
unidirectionnelle batch-first. `forward(input)` donne l’état final ;
`unroll(input,initial_state:null,return_sequence:false)` rend un `GRUResult` :
`state` obligatoire et `sequence` optionnelle. Aucun état ne persiste entre appels.
Les dimensions de features sont positives ; la seed d’initialisation tient dans
uint32. Les paramètres float32 sont construits sur CPU, puis déplacés explicitement.

## Formes et état

L’entrée float32 a la forme `[batch,time,input_features]`. L’état initial float32,
quand il est fourni, a la forme `[batch,hidden_features]` ; sinon il vaut zéro
sur le placement de l’entrée. La sortie finale est `[batch,hidden_features]` ; la
séquence demandée est `[batch,time,hidden_features]`, dans l’ordre temporel.
L’état final égale le dernier élément de la séquence lorsque `time > 0`.
Entrée, état et les quatre paramètres doivent partager placement et device GPU.
Les vues non contiguës sont acceptées, sans fallback ni readback implicite.

Pour `time == 0`, l’état fourni est rendu inchangé, ou un état nul est construit ;
la séquence demandée a la forme `[batch,0,hidden_features]`. Entrée et paramètres
inutilisés restent sans lien autograd. Pour leur demander explicitement un gradient
mathématique nul, employer `Autograd.grad(...,allow_unused:true)`. Le gradient d’un
état initial suivi reste celui de l’opération identité. Un batch nul avec des pas
non nuls conserve les formes et les gradients nuls des opérations exécutées.
Les diagnostics de dtype, forme et placement s’appliquent aussi au temps nul.

L’appelant peut transmettre `result.state` au morceau suivant. Tant que cet état
reste suivi, sorties et gradients correspondent à une séquence entière. Après
consommation d’un graphe, recalculer le forward ; employer explicitement `detach()`
si l’intention est de couper les gradients entre morceaux. La couche ne détache
jamais automatiquement l’état.

## Portes et paramètres

Cette GRU adopte la convention **reset-after**, portes `r,z,n` :

```text
r = sigmoid(x Wr + bir + h Ur + bhr)
z = sigmoid(x Wz + biz + h Uz + bhz)
n = tanh(x Wn + bin + r * (h Un + bhn))
h_next = (1 - z) * n + z * h
```

`parameters()` rend, dans cet ordre, `name.input_weight` de forme
`[input_features,3*hidden_features]`, `name.recurrent_weight` de forme
`[hidden_features,3*hidden_features]`, puis `name.input_bias` et
`name.recurrent_bias` de forme `[3*hidden_features]`. Les colonnes groupées sont
`r`, puis `z`, puis `n` ; les poids sont row-major `[entrée,sortie]`. Les matrices
emploient une initialisation Xavier uniforme, les biais zéro. Une conversion de
poids doit respecter cet ordre, ces orientations et la convention reset-after ;
les deux biais restent distincts, car le biais récurrent de `n` est multiplié par
`r`. Une autre convention de portes n’est pas une simple conversion de noms.
Les cardinalités et tailles en octets sont contrôlées avant allocation.

La couche participe au [registre récursif](Registration.md) : quatre paramètres,
aucun buffer ni enfant, partage par identité, placement et gel. `train()`/`eval()`
se propagent, sans changer ce calcul dépourvu de dropout. La seed ne sert qu’à
l’initialisation, sans compteur aléatoire runtime.

Le [checkpoint complet](TrainingCheckpoint.md) restaure paramètres, mode,
optimiseur et lots ; la configuration inclut dimensions et convention de portes.
L’état récurrent explicite appartient à l’application et n’est pas sauvegardé
comme un état caché de GRU. Le checkpoint des seuls paramètres conserve son contrat.

## Exemple public complet

Ce modèle apprend à classer six séquences asymétriques dont le dernier token
est identique : il doit exploiter les pas précédents. Les quarante pas AdamW et
les profils CPU/Metal qualifient cet exemple synthétique uniquement. La reprise
exacte est testée sur le même backend ; aucune trajectoire bit-identique entre
CPU et Metal n’est promise.

```sx
use Tensor
use Tensor.Neural
use Tensor.Data
use Tensor.Optim
use Tensor.Profile
use STD.Math
use GFX.GPU

nocopy class GRUClassifier:Neural.Layer {
    private var memory:Neural.GRU
    private var classifier:Neural.Dense
    init(seed:int,hidden_features:int = 4) {
        self.memory=Neural.GRU("tiny.memory",2,hidden_features,seed)
        self.classifier=Neural.Dense("tiny.classifier",hidden_features,3,seed+1)
        self.register_child("memory",self.memory)
        self.register_child("classifier",self.classifier)
    }
    override func forward(input:Tensor) Tensor {return self.classifier.forward(self.memory.forward(input))}
}
func gru_training_source() Data.TensorSource {
    var samples:float[]=[0.7,0.1,0.3,0.2,0.05,0.1,0.9,0.2,0.4,0.1,0.05,0.1,-0.5,0.4,-0.3,0.6,0.05,0.1,-0.7,0.3,-0.4,0.5,0.05,0.1,0.1,-0.8,-0.2,-0.5,0.05,0.1,0.2,-0.6,-0.1,-0.4,0.05,0.1]
    var targets:int32[]=[0,0,1,1,2,2]
    return Data.TensorSource(Tensor(samples,[6,3,2]),Tensor(targets,[6]))
}
func gru_learns(device:GPU.Device?) {
    var model=GRUClassifier(seed:17);var source=gru_training_source();var cursor=Data.BatchCursor(source,6,shuffle:false)
    var batch=try cursor.next() else error {panic(error)}
    if var value=batch {
        if let gpu=device {model.to(gpu);value.input=value.input.to(gpu);value.target=value.target.to(gpu)}
        let initial=model.forward(value.input).cross_entropy(value.target,1).detach().cpu().values()[0]
        var optimizer=Optim.AdamW(model.parameters(),learning_rate:0.04)
        var step=0
        while step<40 {
            var profile:Profile?=null
            if let gpu=device {profile=Profile(gpu)}
            model.forward(value.input).cross_entropy(value.target,1).backward();optimizer.step();optimizer.zero_grad()
            if let p=profile {assert(p.snapshot().uploads==0 && p.snapshot().downloads==0)}
            step++
        }
        let final=model.forward(value.input).cross_entropy(value.target,1).detach().cpu().values()[0]
        print("GRU synthetic loss: ",initial," -> ",final)
        assert(Math.is_finite(final) && final<initial*0.25)
    } else {panic("empty GRU learning source")}
}

func main() {
    gru_learns(null)
    if GPU.Device.is_supported() {var device=GPU.Device(GPU.DeviceSettings(debug:true));gru_learns(device)}
}
```

Depuis la racine `SilexProject`, lancer
`silex run Packages/Tensor/Tests/Consumer/Examples/TrainGRU.sx`, ou ajouter
`--release`. Les tests ordinaires comparent sorties et six VJP à un
[oracle Decimal60 indépendant](../../Tools/Oracle/GRU_REPORT.json), tolérance
absolue `2e-6`, puis vérifient Jacobiennes, segmentation et reprise exacte.
Une couche bidirectionnelle, empilée avec dropout ou à état implicite n’est pas fournie.
