# Entraîner un petit Transformer causal

`Neural.TransformerBlock` compose attention multi-têtes, deux résidus, deux
LayerNorm et un MLP Dense/ReLU/Dense. Choisissez `norm_first:true` pour pre-norm
ou `false` pour post-norm. Le bloc dérive de Layer : état, partage, gel, mode et
placement suivent le [registre](Registration.md).

Ce programme entraîne un modèle externe sur six séquences de trois tokens.
La cible cyclique locale associe 1→2, 2→3 et 3→1. Embedding et projection de
classification encadrent le bloc ; l’application choisit les positions sin/cos
et la visibilité causale dans deux buffers enregistrés.

```sx
use Tensor
use Tensor.Neural
use Tensor.Data
use Tensor.Optim
use Tensor.Profile
use STD.Math
use STD.File
use STD.FileSystem
use GFX.GPU

nocopy class TinyTransformer:Neural.Layer {
    private var tokens:Neural.Embedding
    private var positions:Neural.Buffer
    private var visibility:Neural.Buffer
    private var block:Neural.TransformerBlock
    private var classifier:Neural.Dense

    init(seed:int, norm_first:bool = true, dropout_probability:float = 0.0) {
        self.tokens=Neural.Embedding("language.tokens",4,4,seed,padding_index:0)
        self.block=Neural.TransformerBlock("language.block",4,2,8,seed+1,norm_first:norm_first,dropout_probability:dropout_probability)
        self.classifier=Neural.Dense("language.classifier",4,4,seed+2)
        var positions:float[]=[]
        var index=0
        while index<3 {
            positions.append(Math.sin(index as float));positions.append(Math.cos(index as float))
            positions.append(Math.sin((index as float)*0.01));positions.append(Math.cos((index as float)*0.01))
            index++
        }
        self.positions=Neural.Buffer("language.positions",Tensor(positions,[1,3,4]))
        self.visibility=Neural.Buffer("language.visibility",Neural.causal_mask(3,3))
        self.register_child("tokens",self.tokens)
        self.register_child("block",self.block)
        self.register_child("classifier",self.classifier)
        self.register_buffer(self.positions);self.register_buffer(self.visibility)
    }
    override func forward(input:Tensor) Tensor {return self.predict(input,0)}
    func predict(input:Tensor,iteration:int) Tensor {
        let embeddings=self.tokens.forward(input).add(self.positions.value())
        return self.classifier.forward(self.block.encode(embeddings,mask:self.visibility.value(),iteration:iteration).output)
    }
}
func transformer_source() Data.TensorSource {
    var ids:int32[]=[1,2,3,2,3,1,3,1,2,1,2,3,2,3,1,3,1,2]
    var next:int32[]=[2,3,1,3,1,2,1,2,3,2,3,1,3,1,2,1,2,3]
    return Data.TensorSource(Tensor(ids,[6,3]),Tensor(next,[6,3]))
}
func transformer_learns(device:GPU.Device?) {
    var model=TinyTransformer(17)
    var cursor=Data.BatchCursor(transformer_source(),6,shuffle:false)
    if var batch=try cursor.next() else error {panic(error)} {
        if let gpu=device {model.to(gpu);batch.input=batch.input.to(gpu);batch.target=batch.target.to(gpu)}
        model.eval()
        let initial=model.forward(batch.input).cross_entropy(batch.target,2).detach().cpu().values()[0]
        var optimizer=Optim.AdamW(model.parameters(),learning_rate:0.04)
        model.train()
        var step=0
        while step<30 {
            var profile:Profile?=null;if let gpu=device {profile=Profile(gpu)}
            model.predict(batch.input,step).cross_entropy(batch.target,2).backward();optimizer.step();optimizer.zero_grad()
            if let p=profile {assert(p.snapshot().uploads==0 && p.snapshot().downloads==0)}
            step++
        }
        model.eval()
        let final=model.forward(batch.input).cross_entropy(batch.target,2).detach().cpu().values()[0]
        print("Tiny Transformer loss: ",initial," -> ",final)
        assert(Math.is_finite(final) && final<initial*0.4)
    } else {panic("empty Transformer learning source")}
}
func main() {
    transformer_learns(null)
    if GPU.Device.is_supported() {var device=GPU.Device(GPU.DeviceSettings(debug:true));transformer_learns(device)}
}
```

Le programme affiche la loss avant et après trente pas AdamW et vérifie sa
baisse sur CPU, puis sur Metal si disponible. Le test porte sur cette petite
cible synthétique ; il ne qualifie ni tokenizer ni modèle préentraîné. Les
logits ont forme `[batch,sequence,vocabulary]`, les cibles int32
`[batch,sequence]` : `cross_entropy(target,2)` conserve ces formes directement.
Les transferts précèdent l’entraînement. Chaque pas vérifie qu’aucun upload ou
readback n’est ajouté par le forward, backward ou optimiseur.

## Choisir où normaliser

Pour `norm_first:true`, le bloc calcule `r=x+dropout(attention(norm1(x)))`,
puis `y=r+dropout(MLP(norm2(r)))`. Pour `false`, il calcule
`r=norm1(x+dropout(attention(x)))`, puis `y=norm2(r+dropout(MLP(r)))`.
LayerNorm utilise la variance de population sur les features, puis ajoute
epsilon avant la racine carrée. Sur GPU float32, cette paire est fusionnée en
une passe ; moyenne, variance et dérivation conservent leur ordre. Les preuves
et les limites des gains sont décrites dans le [profilage](Profiling.md).

La visibilité suit les [masques d’attention](Attention.md) : `true` autorise,
padding et causalité sont explicites. Elle ne porte que sur la branche attention.
Une query entièrement masquée conserve donc son résidu et la contribution du
MLP ; elle n’est pas automatiquement une position de sortie nulle.
L’encodage positionnel est ajouté par l’application avant le bloc, comme ici.

## Appeler le bloc et reprendre

`TransformerBlock(name,features,heads,feed_forward_features,seed,
norm_first:true,dropout_probability:0.0,epsilon:0.00001)` exige des dimensions
positives, features divisible par heads, seed uint32, probabilité finie dans
`[0,1)` et epsilon fini strictement positif. L’entrée float32 a forme
`[batch,sequence,features]`, batch et séquence éventuellement nuls. Paramètres,
entrée et masque partagent placement et device ; aucun fallback n’est ajouté.
Valeurs et scores visibles doivent rester finis en float32.

`encode(input,mask:null,iteration:0,return_weights:false)` rend un
`TransformerResult` avec `output` et `weights` optionnels. La sortie garde la
forme de l’entrée ; les poids `[batch,heads,sequence,sequence]` sont ceux de
l’attention après son dropout. Les deux dropouts résiduels ne changent pas ces
poids. `forward(input)` fournit le raccourci Layer sans masque à itération0.

L’itération uint32 choisit explicitement les tirages ; un appel répété sur le
même backend reproduit le résultat. Des seeds dérivées de la seed de construction
séparent attention et branches résiduelles. En évaluation, les trois dropouts
sont désactivés. Pour entraîner avec dropout, utilisez `encode` et transmettez
`iteration:progress.step`, comme l’appel `predict` du modèle ci-dessus.

Les quatre enfants sont `attention`, `attention_norm`, `feed_forward_norm` et
`feed_forward`. Les seize paramètres suivent cet ordre : huit de l’attention,
puis weight/bias des deux norms et des Dense `name.feed_forward.input` et
`name.feed_forward.output`. Les matrices Dense sont `[input,output]` ; normes
et biais suivent l’ordre des features. Une instance partagée conserve une
collection unique et une mise à jour par paramètre.

Le [checkpoint complet](TrainingCheckpoint.md) conserve paramètres, buffers,
modes, seeds des trois dropouts, moments d’optimiseur et progression/curseur.
Il refuse une convention pre/post-norm incompatible avant mutation. La reprise
GPU transfère explicitement modèle et moments comme dans ce guide. Un checkpoint
de paramètres seul garde son rôle distinct. Les tests de ce modèle qualifient
une reprise exacte après trois pas puis cinq suivants, avec préchargement CPU.
