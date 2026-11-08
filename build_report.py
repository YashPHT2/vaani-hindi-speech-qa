"""Render the scientific Markdown source as a print-ready, vector PDF.

Usage: python build_report.py [--draft] [--output path.pdf]
Windows fonts are auto-discovered; set VAANI_FONT and VAANI_FONT_BOLD elsewhere.
"""
from __future__ import annotations
import argparse
import html
import os
import re
from pathlib import Path

from markdown_it import MarkdownIt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Flowable, KeepTogether, BaseDocTemplate, Frame, PageTemplate)

ROOT = Path(__file__).resolve().parent
NAVY = colors.HexColor('#17324D')
TEAL = colors.HexColor('#087E8B')
GRAY = colors.HexColor('#526171')
WIDTH = 483.3


def register_fonts():
    regular = Path(os.environ.get('VAANI_FONT', r'C:\Windows\Fonts\Nirmala.ttc'))
    bold = Path(os.environ.get('VAANI_FONT_BOLD', str(regular)))
    if not regular.exists():
        raise SystemExit('Set VAANI_FONT and VAANI_FONT_BOLD to Unicode TrueType fonts with Devanagari support.')
    pdfmetrics.registerFont(TTFont('Body', str(regular), subfontIndex=3 if regular.suffix == '.ttc' else 0, shapable=True))
    pdfmetrics.registerFont(TTFont('BodyBold', str(bold), subfontIndex=4 if bold.suffix == '.ttc' else 0, shapable=True))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='Body', boldItalic='BodyBold')
    # Keep unshaped canvas labels in a distinct font face from shaped paragraphs.
    # Sharing one face with both shaping modes can corrupt subset glyph mapping.
    canvas_regular = Path(os.environ.get('VAANI_CANVAS_FONT', r'C:\Windows\Fonts\arial.ttf'))
    canvas_bold = Path(os.environ.get('VAANI_CANVAS_FONT_BOLD', r'C:\Windows\Fonts\arialbd.ttf'))
    if canvas_regular.exists() and canvas_bold.exists():
        pdfmetrics.registerFont(TTFont('CanvasText', str(canvas_regular), shapable=False))
        pdfmetrics.registerFont(TTFont('CanvasBold', str(canvas_bold), shapable=False))
    else:
        pdfmetrics.registerFont(pdfmetrics.Font('CanvasText','Helvetica','WinAnsiEncoding'))
        pdfmetrics.registerFont(pdfmetrics.Font('CanvasBold','Helvetica-Bold','WinAnsiEncoding'))


def inline(tokens):
    out = []
    links = []
    for token in tokens or []:
        kind = token.type
        if kind == 'text':
            out.append(html.escape(token.content))
        elif kind == 'code_inline':
            out.append('<font color="#145969">' + html.escape(token.content) + '</font>')
        elif kind == 'strong_open': out.append('<b>')
        elif kind == 'strong_close': out.append('</b>')
        elif kind == 'em_open': out.append('<i>')
        elif kind == 'em_close': out.append('</i>')
        elif kind == 'link_open':
            url = token.attrGet('href')
            if not re.match(r'https?://', url or ''):
                # Relative companion files remain readable labels in print.
                out.append('<font color="#087E8B">')
                links.append('</font>')
            else:
                out.append('<link href="' + html.escape(url, quote=True) + '" color="#087E8B">')
                links.append('</link>')
        elif kind == 'link_close':
            out.append(links.pop())
        elif kind in ('softbreak', 'hardbreak'): out.append(' ')
    return ''.join(out)


class Architecture(Flowable):
    """Code-native schematic preserving each Mermaid node and dependency."""
    def __init__(self, duplex=False):
        Flowable.__init__(self)
        self.width = WIDTH
        self.height = 205 if duplex else 150
        self.duplex = duplex

    def draw(self):
        c = self.canv
        def box(x, y, w, h, label, fill='#EDF5F7'):
            c.setFillColor(colors.HexColor(fill)); c.setStrokeColor(colors.HexColor('#ABC5CF'))
            c.roundRect(x,y,w,h,5,fill=1,stroke=1)
            p=Paragraph(label, ParagraphStyle('node',fontName='Body',fontSize=8.3,leading=11,textColor=NAVY,alignment=1,shaping=True))
            _, ph=p.wrap(w-12,h-4); p.drawOn(c,x+6,y+(h-ph)/2)
        def arrow(points):
            c.setStrokeColor(TEAL);c.setFillColor(TEAL);c.setLineWidth(1)
            p=c.beginPath();p.moveTo(*points[0])
            for pt in points[1:]:p.lineTo(*pt)
            c.drawPath(p)
            x,y=points[-1]; px,py=points[-2]
            import math
            a=math.atan2(y-py,x-px)
            p=c.beginPath();p.moveTo(x,y)
            p.lineTo(x-5*math.cos(a-.5),y-5*math.sin(a-.5))
            p.lineTo(x-5*math.cos(a+.5),y-5*math.sin(a+.5));p.close();c.drawPath(p,fill=1,stroke=0)
        c.setFont('CanvasBold',10);c.setFillColor(NAVY)
        if not self.duplex:
            c.drawString(0,138,'Figure 1. Qwen3-Omni: context to streaming waveform')
            nodes=[(0,85,85,34,'User audio'),(103,85,155,34,'Mel features to AuT embeddings'),(276,85,207,34,'Thinker • text and multimodal context'),(276,25,90,36,'Talker • codec frame'),(140,25,118,36,'MTP • residual codes'),(0,25,122,36,'Causal waveform decoder')]
            for n in nodes:box(*n)
            arrow([(85,102),(103,102)]);arrow([(258,102),(276,102)])
            arrow([(379,85),(379,72),(321,72),(321,61)])
            arrow([(276,43),(258,43)]);arrow([(140,43),(122,43)])
            c.setFont('CanvasText',8);c.drawString(0,8,'Waveform output: reply audio. Vision and implementation detail omitted.')
        else:
            c.drawString(0,193,'Figure 2. PersonaPlex: synchronized dialogue streams')
            for n in [(0,132,102,37,'User audio'),(119,132,159,37,'Mimi codes on a continuous clock'),(0,65,128,40,'Voice sample + text role'),(297,107,186,47,'Temporal transformer'),(296,44,88,40,'Aligned assistant text / padding'),(395,44,88,40,'Depth transformer • audio codes'),(191,0,118,30,'Previous agent codes'),(347,0,136,30,'Mimi decoder to assistant audio')]:box(*n)
            arrow([(102,150),(119,150)]);arrow([(278,150),(297,150)])
            arrow([(128,85),(280,85),(280,120),(297,120)])
            arrow([(340,107),(340,84)]);arrow([(439,107),(439,84)])
            arrow([(439,44),(439,30)]);arrow([(340,44),(340,38),(260,38),(260,30)])
            arrow([(410,44),(410,34),(324,34),(324,15),(309,15)])
            arrow([(191,15),(166,15),(166,120),(297,120)])


def build(source, output, draft):
    register_fonts()
    text=source.read_text(encoding='utf-8').replace('\u2011','-').replace('\u2013','-').replace('\u2014',' - ')
    if not draft and re.search(r'No verified model prediction|Measurement status|\bPENDING\b',text,re.I):
        raise SystemExit('Report still contains pending measurement status. Use --draft only for internal layout review.')
    styles=getSampleStyleSheet()
    body=ParagraphStyle('ReportBody',fontName='Body',fontSize=9.8,leading=13.5,spaceAfter=6,textColor=NAVY,shaping=True)
    h1=ParagraphStyle('ReportTitle',parent=body,fontName='BodyBold',fontSize=26,leading=31,spaceAfter=16)
    h2=ParagraphStyle('ReportHeading',parent=body,fontName='BodyBold',fontSize=15,leading=20,spaceBefore=15,spaceAfter=8,keepWithNext=True)
    cell=ParagraphStyle('Cell',parent=body,fontSize=8.2,leading=11,spaceAfter=0)
    story=[]
    if draft: story.append(Paragraph('INTERNAL DRAFT • MEASUREMENTS PENDING',ParagraphStyle('draft',parent=body,textColor=colors.HexColor('#A34413'),fontName='BodyBold')))
    tokens=MarkdownIt('commonmark').enable('table').parse(text)
    i=0; diagram=0
    while i<len(tokens):
        t=tokens[i]
        if t.type=='heading_open':
            content=tokens[i+1];story.append(Paragraph(inline(content.children),h1 if t.tag=='h1' else h2));i+=3;continue
        if t.type=='paragraph_open':
            story.append(Paragraph(inline(tokens[i+1].children),body));i+=3;continue
        if t.type=='fence':
            if t.info.strip()=='mermaid':story.append(Architecture(duplex=diagram>0));diagram+=1;story.append(Spacer(1,10))
            else: story.append(Paragraph(html.escape(t.content).replace('\n','<br/>'),cell))
        elif t.type=='table_open':
            rows=[];row=[];i+=1
            while tokens[i].type!='table_close':
                if tokens[i].type=='tr_open':row=[]
                elif tokens[i].type=='inline':row.append(Paragraph(inline(tokens[i].children),cell))
                elif tokens[i].type=='tr_close':rows.append(row)
                i+=1
            n=len(rows[0]); widths=[WIDTH/n]*n
            if n==3:widths=[100,(WIDTH-100)/2,(WIDTH-100)/2]
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DCEAF0')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F3F7F9')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#ABC5CF'))]))
            # Keep a compact comparison together rather than leave one row behind.
            _, table_height = table.wrap(WIDTH, 720)
            story.extend([KeepTogether([table]) if table_height < 675 else table,Spacer(1,10)])
        i+=1
    output.parent.mkdir(parents=True,exist_ok=True)
    doc=BaseDocTemplate(str(output),pagesize=(595.3,841.9),leftMargin=56,rightMargin=56,topMargin=56,bottomMargin=54,title='Vaani Research Lab - Speech Architecture and Hindi QA',author='Internship assignment')
    def frame(c,d):
        from fontTools.ttLib import TTFont as OutlineFont
        from fontTools.pens.reportLabPen import ReportLabPen
        outline_font=OutlineFont(os.environ.get('VAANI_CANVAS_FONT',r'C:\Windows\Fonts\arial.ttf'))
        glyphs=outline_font.getGlyphSet(); cmap=outline_font.getBestCmap()
        c.saveState();c.setFont('CanvasText',8);c.setFillColor(GRAY)
        def outlined(text,x,y):
            c.saveState();c.translate(x,y);c.scale(8/outline_font['head'].unitsPerEm,8/outline_font['head'].unitsPerEm)
            for character in text:
                name=cmap[ord(character)];path=c.beginPath();path.closePath=path.close
                pen=ReportLabPen(glyphs,path)
                glyphs[name].draw(pen);c.drawPath(pen.path,fill=1,stroke=0)
                c.translate(outline_font['hmtx'][name][0],0)
            c.restoreState()
        outlined('VAANI RESEARCH LAB  /  INTERNSHIP ASSIGNMENT',56,815)
        c.setStrokeColor(colors.HexColor('#D7E0E5'));c.line(56,804,539,804)
        outlined('Architecture review + reproducible Hindi listening-comprehension study',56,28)
        outlined(str(d.page),539-pdfmetrics.stringWidth(str(d.page),'CanvasText',8),28);c.restoreState()
    doc.addPageTemplates(PageTemplate(id='report',frames=[Frame(56,54,483.3,731.9,leftPadding=6,rightPadding=6,topPadding=6,bottomPadding=6)]))
    doc.build(story)
    # Isolate page furniture from any text/graphics state left by shaped tables.
    import io
    from types import SimpleNamespace
    from reportlab.pdfgen.canvas import Canvas
    from pypdf import PdfReader, PdfWriter
    reader=PdfReader(io.BytesIO(output.read_bytes())); writer=PdfWriter()
    overlay=io.BytesIO(); canvas=Canvas(overlay,pagesize=(595.3,841.9))
    for number in range(1,len(reader.pages)+1):
        frame(canvas,SimpleNamespace(page=number));canvas.showPage()
    canvas.save();overlay.seek(0); overlay_reader=PdfReader(overlay)
    for number,page in enumerate(reader.pages):
        page.merge_page(overlay_reader.pages[number]);writer.add_page(page)
    writer.add_metadata({'/Title':'Vaani Research Lab - Speech Architecture and Hindi QA','/Author':'Anonymous internship assignment'})
    with output.open('wb') as stream:writer.write(stream)
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft',action='store_true')
    parser.add_argument('--source',type=Path,default=ROOT/'report.md')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    build(args.source,args.output or ROOT/('tmp/pdfs/report-draft.pdf' if args.draft else 'output/pdf/vaani-research-assignment.pdf'),args.draft)
