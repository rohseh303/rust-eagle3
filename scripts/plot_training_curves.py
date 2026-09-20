"""Plot recorded validation proxies, without fitting a saturation curve."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / 'phase2/training-curves.json').read_text())
    labels = {'small400': '512 Rust / 400-update schedule',
              'scale400': '2,048 Rust / 400-update schedule',
              'mixed400': '1,536 Rust + 512 general / 400-update schedule'}
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for key, label in labels.items():
        if key not in data:
            continue
        x = sorted(map(int, data[key]))
        y = [data[key][str(step)]['eval/simulated_acc_len'] for step in x]
        ax.plot(x, y, marker='o', label=label)
    ax.set(xlabel='Optimizer updates (8 examples per update)',
           ylabel='Validation simulated acceptance-length proxy',
           title='Recorded validation progress under the same 400-update schedule')
    ax.legend(loc='best')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.15)
    fig.supxlabel('Higher is better on this proxy; it is not measured serving speed or evidence of saturation.', fontsize=9)
    for ext in ('png', 'svg'):
        fig.savefig(ROOT / f'phase2/figures/training-curves.{ext}', dpi=170)


if __name__ == '__main__':
    main()
