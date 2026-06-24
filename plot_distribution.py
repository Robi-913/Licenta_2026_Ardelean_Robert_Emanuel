import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

biomarkers = [
    "Soft Drusen", "Hard Drusen", "PR Layer\nDisruption",
    "Soft Drusen PED", "Reticular\nDrusen", "Geographic\nAtrophy",
    "Choroidal\nFolds", "Hyperfluorescent\nSpots", "Fluid"
]
counts = [359, 303, 209, 97, 95, 72, 64, 33, 14]
total = 566
percentages = [c/total*100 for c in counts]

# culori: verde pentru comune, portocaliu pentru rare (sub 20%)
colors = ['#2ecc71' if p > 20 else '#e67e22' if p > 5 else '#e74c3c'
          for p in percentages]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(biomarkers[::-1], percentages[::-1], color=colors[::-1],
               edgecolor='white', linewidth=0.5)

# adauga valorile
for bar, count, pct in zip(bars, counts[::-1], percentages[::-1]):
    ax.text(pct + 0.5, bar.get_y() + bar.get_height()/2,
            f'{count} ({pct:.1f}%)', va='center', fontsize=10)

ax.set_xlabel('Prevalență (%)', fontsize=12)
ax.set_title('Distribuția biomarkerilor în OCT5k (566 imagini cu adnotări)',
             fontsize=13, fontweight='bold')
ax.set_xlim(0, 80)
ax.axvline(x=5, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=20, color='orange', linestyle='--', alpha=0.5, linewidth=1)

# legenda
patches = [
    mpatches.Patch(color='#2ecc71', label='Frecvent (>20%)'),
    mpatches.Patch(color='#e67e22', label='Rar (5-20%)'),
    mpatches.Patch(color='#e74c3c', label='Foarte rar (<5%)'),
]
ax.legend(handles=patches, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('biomarker_distribution.pdf', bbox_inches='tight', dpi=300)
plt.savefig('biomarker_distribution.png', dpi=300)
print("Salvat in figs/biomarker_distribution.pdf")