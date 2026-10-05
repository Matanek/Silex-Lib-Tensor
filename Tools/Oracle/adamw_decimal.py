"""Development-only mathematical AdamW oracle; ordinary tests use pinned literals."""
from decimal import Decimal, localcontext
from pathlib import Path
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
gradients = [["0.3", "-0.2"], ["-0.1", "0.4"], ["0", "0"],
             ["1", "-2"], ["0.0001", "-0.0002"], ["0.3", "0.1"]]
with localcontext() as context:
    context.prec = 50
    values = [Decimal("1"), Decimal("-2")]
    first = [Decimal(0), Decimal(0)]
    second = [Decimal(0), Decimal(0)]
    rows = []
    for step, gradient in enumerate(gradients, 1):
        for index, g in enumerate(map(Decimal, gradient)):
            first[index] = Decimal(".8") * first[index] + Decimal(".2") * g
            second[index] = Decimal(".9") * second[index] + Decimal(".1") * g * g
            numerator = first[index] / (1 - Decimal(".8") ** step)
            denominator = (second[index] / (1 - Decimal(".9") ** step)).sqrt() + Decimal(".000001")
            values[index] = values[index] * (1 - Decimal(".01") * Decimal(".05")) - Decimal(".01") * numerator / denominator
        rows.append(dict(step=step, parameter=list(map(str, values)),
                         first_moment=list(map(str, first)), second_moment=list(map(str, second))))
report = dict(oracle="Decimal 50 digits, mathematical AdamW recurrence, no Tensor implementation",
              initial=["1", "-2"], gradients=gradients, learning_rate="0.01", beta1="0.8",
              beta2="0.9", epsilon="0.000001", weight_decay="0.05", steps=rows)
path = Path(__file__).with_name("ADAMW_REPORT.json")
content = json.dumps(report, indent=2) + "\n"
if args.check:
    if path.read_text() != content:
        raise SystemExit("AdamW report differs from the independent Decimal oracle")
else:
    path.write_text(content)
