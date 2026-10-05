# Propager le mode, le placement et le gel d’un modèle

Le registre de Layer permet de changer l’état de tout le modèle par une seule
instruction : `train()`, `eval()`, `to(device)`, `cpu()`, `freeze()`, `unfreeze()`
et `zero_grad()`. Les identités enregistrées partagées sont traitées une fois,
y compris lorsque deux branches rejoignent la même couche.

Ce programme compte les notifications de la couche partagée. Il qualifie aussi
les transferts explicites avec un GPU compatible GFX.GPU :

```sx
use Tensor
use Tensor.Neural
use Tensor.Optim
use GFX.GPU

nocopy class StateModel:Neural.Layer {
    init() {}
    func attach(name:str, child:Neural.Layer) { self.register_child(name,child) }
    func add_parameter(parameter:Neural.Parameter) { self.register_parameter(parameter) }
    func add_buffer(buffer:Neural.Buffer) { self.register_buffer(buffer) }
    override func forward(input:Tensor) Tensor { return input }
}

nocopy class CountedLayer:Neural.Layer {
    private var mode_calls:int
    private var gpu_calls:int
    private var cpu_calls:int
    init(weight:Neural.Parameter, scale:Neural.Buffer) {
        self.mode_calls = 0; self.gpu_calls = 0; self.cpu_calls = 0
        self.register_parameter(weight); self.register_buffer(scale)
    }
    override func training_changed(training:bool) {
        assert(self.is_training() == training)
        self.mode_calls++
    }
    override func moved_to(device:GPU.Device) { self.gpu_calls++ }
    override func moved_to_cpu() { self.cpu_calls++ }
    func calls() int[] { return [self.mode_calls,self.gpu_calls,self.cpu_calls] }
    override func forward(input:Tensor) Tensor {
        return input.multiply(self.parameters()[0].value()).multiply(self.buffers()[0].value())
    }
}

func main() {
    var weight = Neural.Parameter("weight",Tensor.scalar(2.0)); var scale = Neural.Buffer("scale",Tensor.scalar(3.0))
    var leaf = CountedLayer(weight,scale)
    var first = StateModel(); var second = StateModel(); var root = StateModel()
    first.attach("shared",leaf); second.attach("shared",leaf); root.attach("first",first); root.attach("second",second)
    root.eval(); assert(!leaf.is_training() && leaf.calls()[0] == 1)
    root.freeze(); assert(!weight.is_trainable())
    root.unfreeze(); assert(weight.is_trainable())
    assert(GPU.Device.is_supported())
    var device = GPU.Device(GPU.DeviceSettings(debug:true))
    root.to(device); assert(weight.value().is_gpu() && scale.value().is_gpu() && leaf.calls()[1] == 1)
    root.cpu(); assert(weight.value().is_cpu() && scale.value().is_cpu() && leaf.calls()[2] == 1)
}
```

Chaque appel du modèle donne une notification locale au leaf, malgré ses deux
chemins. Le coefficient Buffer et le poids se déplacent ensemble. Le gel ne
change ni leurs valeurs ni l’identité du poids ; le Buffer reste non entraînable.

## Définir un hook local

Dans le style enregistré, héritez des méthodes publiques de parcours et redéfinissez
seulement les notifications protégées qui vous sont utiles :

- `training_changed(training:bool)` reçoit le mode après sa mise à jour ;
- `moved_to(device:GPU.Device)` est appelé après déplacement de l’état sur ce GPU ;
- `moved_to_cpu()` est appelé après déplacement sur CPU.

Ces hooks décrivent uniquement l’état local de votre couche. Ne relayez pas les
enfants dedans : le parcours central les visite déjà. Une couche enregistrée ne
redéfinit donc pas train/eval/to/cpu pour récursivement appeler ses enfants.
`is_training()` rend le mode de Layer, initialement `true`. Chaque demande de mode
notifie les couches, même si le mode demandé est déjà établi. Dropout utilise ce
même mode ; son compteur d’itérations ne progresse qu’en entraînement.

Une ancienne couche sans registre conserve ses overrides publics manuels : le
parcours enregistré appelle cette frontière opaque une fois. Les enfants cachés
à l’intérieur de cet override restent sous sa responsabilité. Une couche qui
adopte le registre migre ses hooks manuels vers les notifications locales ;
les opérations héritées prennent alors en charge tout son état déclaré.

## Garder les transferts explicites

La collecte valide les noms avant mutation. `to(device)` et `cpu()` exigent que
les gradients de tous les paramètres collectés soient vides ; appelez d’abord
`model.zero_grad()`. Pour to, les devices de tous les paramètres et buffers GPU
sont contrôlés avant les premiers transferts. Un changement de device passe
explicitement par `model.cpu()` puis `model.to(other_device)`.

Chaque Parameter et Buffer qui suit le modèle est déplacé une fois par identité
enregistrée unique. Les buffers CPU déclarés restent sur CPU. Les buffers
acceptent tous les dtypes de stockage, y compris les booléens, sans conversion.
La collecte et les changements de mode ou de gel n’effectuent aucun readback.
Les paramètres déjà déplacés ne migrent pas automatiquement les moments d’un
optimiseur : les contrôles de cohérence de l’optimiseur restent applicables.

## Geler et reprendre

`freeze()` détache les nouvelles lectures des paramètres du modèle. `unfreeze()`
les rend entraînables. Un changement effectif exige leurs slots vides ; tous les
slots concernés sont vérifiés avant de modifier le premier paramètre. Répéter
l’état déjà établi est sans effet. `zero_grad()` vide tous les slots collectés,
y compris ceux remplis par des graphes antérieurs au gel.

SGD et Adam ignorent les paramètres gelés et conservent leurs moments ; les gradients
des entrées restent calculables. Voir le [contrat individuel de Buffer et Parameter](State.md).
Les [noms, aliases, cycles et identités](Registration.md) définissent la collecte.
Le checkpoint actuel sauvegarde encore les paramètres seuls : buffers, mode,
gel et état d’optimiseur ne font pas partie de son schéma.

[Métadonnées CPU du modèle](HostState.md) : buffers à placement CPU explicite et transferts récursifs.
