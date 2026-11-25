from pptx import Presentation
import os, re

deck = r'D:\StockPredictor\presentation_output\Stock_Market_Intelligence_Engine.pptx'
p = Presentation(deck)
print('SLIDES:', len(p.slides))
print('ASPECT:', round(p.slide_width / p.slide_height, 4), '(expected 1.7778)')
print('SIZE_IN:', round(p.slide_width / 914400, 3), 'x', round(p.slide_height / 914400, 3))

notes_ok = 0
oob = []
for i, s in enumerate(p.slides, 1):
    if s.has_notes_slide and s.notes_slide.notes_text_frame.text.strip():
        notes_ok += 1
    for sh in s.shapes:
        if sh.left is None or sh.top is None or sh.width is None or sh.height is None:
            continue
        if sh.left < -1000 or sh.top < -1000 or sh.left + sh.width > p.slide_width + 1000 or sh.top + sh.height > p.slide_height + 1000:
            oob.append((i, sh.shape_type, round(sh.left/914400,2), round(sh.top/914400,2), round(sh.width/914400,2), round(sh.height/914400,2)))
print('SLIDES_WITH_NOTES:', notes_ok)
print('OUT_OF_BOUNDS_SHAPES:', len(oob))
for o in oob[:10]:
    print(' ', o)

for f in ['presentation_content.md', 'presentation_notes.md']:
    path = os.path.join(os.path.dirname(deck), f)
    print(f, 'EXISTS' if os.path.exists(path) else 'MISSING', os.path.getsize(path) if os.path.exists(path) else '')

renders = [f for f in os.listdir(os.path.join(os.path.dirname(deck), 'render')) if f.endswith('.png')]
print('RENDERED_PNGS:', len(renders))
