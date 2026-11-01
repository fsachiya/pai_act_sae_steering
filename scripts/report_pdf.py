"""docs/report.md → HTML → PDF (Chrome headless). 使い方: uv run --with markdown python scripts/report_pdf.py <出力PDF>"""
import html as htmllib
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
import re

md = (ROOT / "docs/report.md").read_text()
md = re.sub(r"(?m)^(  )+(?=[-*] |\d+\. )", lambda m: "    " * (len(m.group(0)) // 2), md)   # 入れ子の箇条書きを4文字字下げに
md = re.sub(r"(?m)^(?![ \t]*(?:[-*] |\d+\. ))(.+)\n(?=[-*] |\d+\. )", r"\1\n\n", md)   # 段落直後の箇条書きの前に空行を入れる
# 数式は markdown に触らせないよう退避し, 変換後に KaTeX の区切り \[ \] / \( \) で戻す
maths = []


def stash(m, display):
    maths.append(("\\[" + m.group(1) + "\\]") if display else ("\\(" + m.group(1) + "\\)"))
    return f"@@MATH{len(maths) - 1}@@"


md = re.sub(r"\$\$(.+?)\$\$", lambda m: stash(m, True), md, flags=re.S)
md = re.sub(r"(?<![\\$])\$([^$\n]+?)\$", lambda m: stash(m, False), md)
body = markdown.markdown(md, extensions=["tables"])
body = re.sub(r"<p>(@@MATH\d+@@)</p>", r'<div class="math">\1</div>', body)
body = re.sub(r"@@MATH(\d+)@@", lambda m: htmllib.escape(maths[int(m.group(1))], quote=False), body)
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"
html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>説明資料</title>
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [{{left: '\\\\[', right: '\\\\]', display: true}}, {{left: '\\\\(', right: '\\\\)', display: false}}]}})"></script>
<style>
@page {{ size: A4; margin: 16mm 14mm; }}
body {{ font-family: "Hiragino Sans", "Hiragino Kaku Gothic ProN", sans-serif; font-size: 10.5pt; line-height: 1.6; color: #111; }}
h1 {{ font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 4px; }}
h2 {{ font-size: 14pt; margin-top: 1.4em; border-left: 5px solid #2b6cb0; padding-left: 8px; }}
h3 {{ font-size: 12pt; margin-top: 1.1em; }}
table {{ border-collapse: collapse; margin: 0.6em 0; font-size: 9.5pt; }}
th, td {{ border: 1px solid #999; padding: 3px 7px; }} th {{ background: #eef2f7; }}
img {{ max-width: 100%; max-height: 135mm; display: block; margin: 0.5em auto; page-break-inside: avoid; }}
img[alt="刺激した特徴"] {{ max-height: 230mm; }}   /* 縦長の図は幅いっぱいに表示する */
.math {{ margin: 0.6em 0; overflow-x: auto; }}
.katex {{ font-size: 1.05em; }}
code {{ font-size: 9pt; background: #f3f3f3; padding: 0 3px; }}
</style></head><body>{body}</body></html>"""
out_html = ROOT / "docs/report.html"
out_html.write_text(html)
pdf = Path(sys.argv[1]).resolve()
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=20000", f"--print-to-pdf={pdf}", out_html.as_uri()],
               check=True, capture_output=True)
print("saved", pdf)
