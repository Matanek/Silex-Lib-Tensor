# Enregistrer et partager l’état d’une couche

Une classe dérivée de `Neural.Layer` enregistre ses paramètres, buffers et enfants
une fois dans son constructeur. Les méthodes héritées `parameters()` et `buffers()`
collectent cet état récursivement, en conservant chaque identité une seule fois.
Le calcul reste libre : vos méthodes peuvent accepter plusieurs entrées ou sorties.

Ces deux résidus partagent une Dense et un coefficient non entraînable :

```sx
use Tensor
use Tensor.Neural
use Tensor.Optim
use STD.Math

nocopy class RegisteredResidual:Neural.Layer {
    private var block:Neural.Dense
    private var scale:Neural.Buffer
    init(block:Neural.Dense, scale:Neural.Buffer) {
        self.block = block
        self.scale = scale
        self.register_child("block",block)
        self.register_buffer(scale)
    }
    override func forward(input:Tensor) Tensor { return input.add(self.block.forward(input).multiply(self.scale.value())) }
}

func main() {
    var block = Neural.Dense("shared",1,1,seed:0)
    var weights = block.parameters()
    weights[0].replace(Tensor.full([1,1],2.0)); weights[1].replace(Tensor.full([1],1.0))
    var scale = Neural.Buffer("scale",Tensor.scalar(1.0))
    var left = RegisteredResidual(block,scale)
    var right = RegisteredResidual(block,scale)
    var branches:Neural.Layer[] = [left,right]
    var model = Neural.Sequential(branches)
    assert(model.parameters().count() == 2 && model.buffers().count() == 1)
    assert(model.parameters()[0] == weights[0] && model.buffers()[0] == scale)
    var optimizer = Optim.SGD(model.parameters(),learning_rate:0.1)
    left.forward(Tensor.ones([1,1])).add(right.forward(Tensor.ones([1,1]))).sum().backward()
    optimizer.step()
    assert(Math.abs(weights[0].value().item()-1.8) < 0.00001)
    assert(scale.value().item() == 1.0)
}
```

La collecte rend deux paramètres, le poids et le biais de la Dense, et un Buffer.
Chaque branche contribue au gradient du poids ; SGD reçoit pourtant une seule
référence et effectue un seul pas, de 2 à environ 1,8. Le coefficient reste à 1.

## Donner un propriétaire à l’état

Les trois méthodes protégées sont accessibles dans votre classe dérivée :

- `register_parameter(parameter)` enregistre un Parameter existant ;
- `register_buffer(buffer)` enregistre un Buffer existant ;
- `register_child(name, child)` enregistre une couche sous un nom local non vide.

Appelez-les après avoir initialisé les champs de votre classe. Le registre conserve
les identités existantes et leur durée de vie ; il ne clone ni couche ni stockage.
Répéter le même enregistrement est sans effet. Le même enfant peut avoir plusieurs
alias locaux. Un alias existant ne peut pas désigner une autre couche.

Les noms de Parameter et Buffer restent ceux fournis à leur construction : aucun
préfixe implicite ne les modifie. Ils sont uniques dans l’état collecté du modèle,
y compris entre paramètres et buffers. Deux références au même objet sont permises ;
deux objets différents portant le même nom sont refusés. Nommez donc les instances
indépendantes, par exemple `encoder.weight` et `decoder.weight`.

## Inspecter le modèle partagé

`children()` rend les enfants directs par ordre d’enregistrement et identité unique.
`child(name)` rend la couche de cet alias local, ou `null` lorsque le nom est absent.
`parameters()` et `buffers()` rendent les objets propres avant ceux des enfants,
par parcours en profondeur dans l’ordre d’enregistrement. Une couche déjà visitée
n’est pas collectée une seconde fois dans un diamant partagé.

Les listes retournées peuvent être modifiées localement sans changer le registre.
Les objets dedans gardent leur identité partagée. La collecte est recalculée : un
état ajouté ultérieurement à un enfant est visible au prochain appel. Si cet ajout
crée une collision dans un parent, sa prochaine collecte la refuse. Les cycles
visibles dans les enfants enregistrés sont refusés avant l’ajout de l’arête.

Dense, Conv2D, LayerNorm, SimpleRNN et Sequential utilisent cette collecte sans
changer leurs noms existants. Sequential accepte plusieurs occurrences de la même
couche, l’exécute à chaque occurrence dans le forward et collectera son état une fois.
Le checkpoint actuel conserve son schéma de paramètres : un paramètre partagé y
apparaît une fois ; Buffer et état de gel restent hors de ce format.

## Conserver les hooks explicites

Une couche qui adopte le registre laisse `parameters()` et `buffers()` hérités.
Les anciennes couches sans enregistrement peuvent conserver un override manuel :
leur liste est lue comme une frontière opaque. Le registre ne découvre pas les
enfants qu’un tel override conserve uniquement dans ses champs.

L’enregistrement automatise ici la collecte. `train`, `eval`, `to(device)` et `cpu()`
restent des hooks à relayer explicitement dans une couche personnalisée. Les
transferts du modèle GPU de l’exemple s’écrivent donc `block.to(device)` et
`scale.to(device)` avant le forward ; aucun transfert n’est déclenché par la collecte.
Le [guide de composition](Composition.md) décrit ces hooks et le [guide des buffers
et du gel](State.md) décrit leur état individuel.
