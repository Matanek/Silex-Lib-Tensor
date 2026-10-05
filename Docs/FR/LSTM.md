# Entraîner une LSTM avec deux états explicites

`Neural.LSTM(name,input_features,hidden_features,seed)` déroule une couche
unidirectionnelle batch-first. `forward(input)` rend l’état caché final Tensor.
`unroll(input,initial_state:null,return_sequence:false)` rend `LSTMResult` :
`state` est une paire `LSTMState(hidden:Tensor,cell:Tensor)` et `sequence` est
optionnelle. Aucun état ne persiste entre appels. Les dimensions sont positives,
la seed tient dans uint32 ; l’initialisation float32 se fait sur CPU.

## Formes et transmission

L’entrée float32 est `[batch,time,input_features]`. Les deux états initiaux
float32 sont `[batch,hidden_features]`. Fournir la paire complète ou laisser
`initial_state:null` pour deux zeros sur le placement de l’entrée. États finaux
et séquence cachée demandée ont respectivement `[batch,hidden_features]` et
`[batch,time,hidden_features]`. Le hidden final égale le dernier élément de la
séquence quand `time > 0`. La séquence des cellules n’est pas retournée.
Entrée, hidden, cell et les quatre paramètres doivent partager placement/device
GPU. Les vues non contiguës et leurs gradients sont pris en charge, sans fallback
ou readback implicite. Les paramètres se déplacent via les méthodes de Layer.

Transmettre `result.state` au morceau suivant conserve les deux liens de gradient.
Sorties, états et sept VJP correspondent alors au déroulement entier. La couche
ne détache jamais un état automatiquement. Après consommation, recalculer le
forward ; détacher explicitement hidden et cell seulement si l’intention est de
couper les gradients entre morceaux. Transmettre uniquement hidden perdrait la
mémoire portée par cell.

Pour `time == 0`, la paire fournie est rendue inchangée, ou deux états nuls sont
créés ; la séquence optionnelle est `[batch,0,hidden_features]`. Les gradients
d’états suivis sont ceux de l’identité, tandis qu’entrée et paramètres inutilisés
restent sans lien autograd. `Autograd.grad(...,allow_unused:true)` permet de leur
demander un zéro mathématique explicite. Un batch vide avec des pas non nuls garde
ses formes et les gradients nuls des opérations exécutées. Les diagnostics des
deux états et des placements s’appliquent aussi au temps nul.

## Portes et poids

Les colonnes groupées suivent `i,f,g,o` :

```text
a = x W + h U + input_bias + recurrent_bias
i = sigmoid(a_i)
f = sigmoid(a_f)
g = tanh(a_g)
o = sigmoid(a_o)
cell_next = f * cell + i * g
hidden_next = o * tanh(cell_next)
```

`parameters()` retourne dans cet ordre `name.input_weight`
`[input_features,4*hidden_features]`, `name.recurrent_weight`
`[hidden_features,4*hidden_features]`, puis `name.input_bias` et
`name.recurrent_bias` `[4*hidden_features]`. Les matrices sont row-major
`[entrée,sortie]`, Xavier uniforme ; les deux biais commencent à zéro, y compris
la porte d’oubli. Convertir explicitement toute autre orientation ou ordre des
portes avant `Parameter.replace`. Les deux biais restent des paramètres séparés.
Les cardinalités et tailles en octets sont contrôlées avant toute allocation.
Pas de peephole, projection, bidirectionnel intégré ou dropout inter-couches.

La couche rejoint le [registre récursif](Registration.md) avec quatre paramètres,
aucun buffer/enfant, partage par identité et gel. `train()`/`eval()` se propagent,
sans modifier ce calcul sans dropout. La seed sert uniquement à l’initialisation,
sans compteur aléatoire runtime.

Le [checkpoint complet](TrainingCheckpoint.md) conserve paramètres, mode,
optimiseur et lots ; sa configuration inclut dimensions et portes. La paire
récurrente appartient à l’application : elle n’est pas enregistrée comme état
caché de LSTM. Le checkpoint des seuls paramètres conserve son contrat.

## Exemple public complet

Le dernier token des six séquences est identique ; les classes dépendent des
premiers pas. Le modèle ci-dessous utilise une LSTM et un classifieur Dense,
six paramètres enregistrés et quarante pas AdamW. Les profils vérifient les pas
résidents CPU/Metal ; il s’agit d’une qualification synthétique locale.
La reprise exacte est testée sur le même backend, sans promesse de trajectoire
bit-identique entre CPU et Metal.

```sx
use Tensor
use Tensor.Neural
use Tensor.Data
use Tensor.Optim
use Tensor.Profile
use STD.Math
use GFX.GPU

nocopy class LSTMClassifier:Neural.Layer {
    private var memory:Neural.LSTM
    private var classifier:Neural.Dense
    init(seed:int,hidden_features:int = 4) {
        self.memory=Neural.LSTM("tiny.memory",2,hidden_features,seed)
        self.classifier=Neural.Dense("tiny.classifier",hidden_features,3,seed+1)
        self.register_child("memory",self.memory)
        self.register_child("classifier",self.classifier)
    }
    override func forward(input:Tensor) Tensor {return self.classifier.forward(self.memory.forward(input))}
}
func lstm_training_source() Data.TensorSource {
    var samples:float[]=[0.7,0.1,0.3,0.2,0.05,0.1,0.9,0.2,0.4,0.1,0.05,0.1,-0.5,0.4,-0.3,0.6,0.05,0.1,-0.7,0.3,-0.4,0.5,0.05,0.1,0.1,-0.8,-0.2,-0.5,0.05,0.1,0.2,-0.6,-0.1,-0.4,0.05,0.1]
    var targets:int32[]=[0,0,1,1,2,2]
    return Data.TensorSource(Tensor(samples,[6,3,2]),Tensor(targets,[6]))
}
func lstm_learns(device:GPU.Device?) {
    var model=LSTMClassifier(seed:17);var source=lstm_training_source();var cursor=Data.BatchCursor(source,6,shuffle:false)
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
        print("LSTM synthetic loss: ",initial," -> ",final)
        assert(Math.is_finite(final) && final<initial*0.25)
    } else {panic("empty LSTM learning source")}
}

func main() {
    lstm_learns(null)
    if GPU.Device.is_supported() {var device=GPU.Device(GPU.DeviceSettings(debug:true));lstm_learns(device)}
}
```

Depuis `SilexProject`, lancer
`silex run Packages/Tensor/Tests/Consumer/Examples/TrainLSTM.sx`, ou ajouter
`--release`. Les tests hermétiques confrontent sortie/séquence et sept VJP à un
[oracle Decimal60 indépendant](../../Tools/Oracle/LSTM_REPORT.json), tolérance
absolue `2e-6`, puis aux Jacobiennes et à la segmentation explicite.
