# Offline results viewer

`Planetary_Sensing_Viewer.html` is the whole viewer: one file, no internet needed, opens in any current browser. It shows pre-computed results; it never runs the simulation.

To rebuild it after new results:

```bash
python viewer/build_viewer.py
```

The builder reads the trial records of both structures — the habitat floor, the main case (`outputs/raw/records_S2/*.npz`), and the reference bridge, the earlier study's bridge model (`outputs/raw/records/*.npz`); repetition 0 of each trial type is animated, and the floor's walks are paired with the deck's rigid-ground twins cut to the floor's crossing — plus `outputs/tables/*.json` (the floor's `*_S2.json`, `occupied_curves.json`, `hsi_structures.json`, `phaseK_check.json` and the deck's tables) and `outputs/gait/*.json`, quantises the signals to 16-bit integers, and inlines them with the page's own code (`viewer/src/`: `index.html`, `style.css`, `charts.js`, `scene.js` (flat drawing: the reference bridge over a ravine, or the floor in a section through the habitat vessel), `scene3d.js` (3-D scene), `app.js`, `results.js`). The 3-D scene uses three.js r149 (MIT licence, `viewer/vendor/`), inlined so that the file stays offline.

What-if gaits on both structures. The what-if gaits (the Earth gait tuned for 1.25 m/s, scaled to Mars and the Moon by dynamic similarity) are simulated on the habitat floor and on the reference bridge, so the viewer shows them on both (Phase O, `experiments/phase_O_consistency/whatif_floor_trials.py`).

Look: the page is one sheet of an engineering drawing set, drawn in a double border. On the left: the drawing number PS-01, the title, a to-scale bar of the three gravities (Moon 1.62, Mars 3.71, Earth 9.81 m/s²), the sheet index (S-1 Watch a trial, S-2 What gravity changes, S-3 Can the structure be detected?, S-4 Model and checks) and the symbols used on every sheet. At the bottom, a title block: project, sheet number and title, number of sheets, trials and person shown, print, build record and notes. The options of a trial form a selection schedule with round radio markers; the key numbers of a trial are two schedule tables (quantity, value, unit, remark). Each section heading carries a detail bubble (section number over sheet number). Findings are hexagon keynotes (N1, N2, …); caveats are general notes numbered in red triangles. Charts have open axes with outward ticks and hairline grid lines, and their circled detail number and title sit below the chart. Line charts label each line at its right end (with leader lines where labels would touch); a reference line is labelled on the line, at the left when the right end is taken by line labels. Bar charts name their reference lines in the key above the chart, with a different dash for each, because the bars fill the plot. Two prints: whiteprint (navy ink on cool white, the default unless the computer is set to dark mode) and blueprint (pale ink on Prussian blue), chosen in the title block and remembered by the browser; switching recolours the charts in place.

The setting drawing on S-4 is stored as `viewer/assets/setting.png`.

Lettering: B612 and B612 Mono, the typefaces designed for aircraft cockpit displays (Copyright 2012 The B612 Project Authors, SIL Open Font License 1.1; `viewer/fonts/OFL-B612.txt`, `OFL-B612Mono.txt`), embedded in the page so that it looks the same offline everywhere. Numbers use the proportional B612, because the mono face spaces decimal points widely.

Colours (whiteprint / blueprint): ink `#14284b` / `#eaf2ff`; Earth `#1f63c6` / `#8fd3ff`, Mars `#c4511c` / `#ffb27a`, Moon `#646b74` / `#c9d3df`; watch `#0d7f70` / `#7ff0d8`, phone `#97368f` / `#ffa8f0`; amber for thresholds and occupied frequencies, red for markers and failures. Each planet also keeps its own line pattern (solid, dashed, dash-dot) and glyph, so that no information rests on colour alone. The 3-D view and the flat drawing are illustrations inside the viewport and keep their own lighting in both prints.
