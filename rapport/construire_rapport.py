"""Une commande : python construire_rapport.py [--figures] [--controle]."""
from pathlib import Path
import re, sys, os, json, argparse, subprocess
ROOT=Path(__file__).resolve().parent
sys.path.append(str(ROOT/'outils'))
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Table,TableStyle,
                               Image,PageBreak,KeepTogether,Preformatted,Flowable,CondPageBreak)
from reportlab.platypus.tableofcontents import TableOfContents

W,H=A4; M=53; CONTENT=W-2*M
PAL={'audio':'#A36224','texte':'#16766D','video':'#365F94','fusion':'#795071','commun':'#365363'}
PALE={'audio':'#F8F0E5','texte':'#EAF5F1','video':'#EDF2F9','fusion':'#F4EDF3','commun':'#F1F4F6'}
INK=colors.HexColor('#213442'); MUTED=colors.HexColor('#63717B')
for family,stem in [('Texte','constan'),('Sans','calibri')]:
    for suffix,name in [('',family),('b',family+'-B'),('i',family+'-I'),('z',family+'-BI')]:
        # Les variantes grasses italiques de Constantia sont nommées constanz.
        pdfmetrics.registerFont(TTFont(name,str(ROOT/'polices'/(stem+suffix+'.ttf'))))
    pdfmetrics.registerFontFamily(family,normal=family,bold=family+'-B',italic=family+'-I',boldItalic=family+'-BI')

def sty(name,**kw):
    args=dict(fontName='Texte',fontSize=10.25,leading=13.7,textColor=INK,spaceAfter=5.3,
              allowWidows=0,allowOrphans=0,splitLongWords=True)
    args.update(kw);return ParagraphStyle(name,**args)
BODY=sty('body')
CAPTION=sty('caption',fontName='Sans',fontSize=9,leading=11.6,textColor=MUTED,spaceAfter=10)
CELL=sty('cell',fontName='Sans',fontSize=9,leading=11.2,spaceAfter=0)
CODE=sty('code',fontName='Sans',fontSize=8.5,leading=11,spaceAfter=0)

def inline(t):
    t=t.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    t=re.sub(r'`([^`]+)`',r'<font face="Sans" color="#526778">\1</font>',t)
    t=re.sub(r'\*\*([^*]*?)\*([^*]+)\*\*\*',r'<b>\1<i>\2</i></b>',t)
    t=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',t)
    t=re.sub(r'(?<![\w*])\*([^*\n]+)\*(?![\w*])',r'<i>\1</i>',t)
    t=re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',t)
    return t.replace('ᵉ','<super>e</super>')

class Chapter(Paragraph):
    def __init__(self,title,mod='commun'):
        self.chapterTitle=title;self.mod=mod
        super().__init__(inline(title),sty('chapter',fontName='Sans-B',fontSize=24,leading=27,
           textColor=colors.HexColor(PAL[mod]),spaceBefore=8,spaceAfter=15,keepWithNext=1))
    def draw(self):
        self.canv.setFillColor(colors.HexColor(PAL[self.mod]));self.canv.rect(-M,0,5,self.height,fill=1,stroke=0)
        super().draw()

class Banner(Flowable):
    def __init__(self,label,text,mod):
        Flowable.__init__(self);self.label=label;self.mod=mod
        self.para=Paragraph(inline(text),sty('boxtext',fontName='Sans',fontSize=10.3,leading=13.2,spaceAfter=0))
        self.spaceBefore=4;self.spaceAfter=13
    def wrap(self,w,h):
        self.width=w;_,self.ph=self.para.wrap(w-29,h);self.height=self.ph+36;return w,self.height
    def draw(self):
        c=self.canv;c.setFillColor(colors.HexColor(PALE[self.mod]));c.roundRect(0,0,self.width,self.height,5,fill=1,stroke=0)
        c.setFillColor(colors.HexColor(PAL[self.mod]));c.rect(0,0,3,self.height,fill=1,stroke=0)
        c.setFont('Sans-B',9);c.drawString(14,self.height-16,self.label.upper())
        self.para.drawOn(c,14,11)

class Cover(Flowable):
    def __init__(self):Flowable.__init__(self);self.width=CONTENT;self.height=H-110
    def draw(self):
        c=self.canv
        # Coordonnées de page, pour une couverture indépendante du flux courant.
        c.saveState();c.translate(-M,-(H-58-self.height))
        c.setFillColor(colors.HexColor('#F4F2ED'));c.rect(0,0,W,H,fill=1,stroke=0)
        c.setFillColor(colors.white);c.rect(0,H-154,W,154,fill=1,stroke=0)
        c.drawImage(str(ROOT/'logos/universite_toulouse3.png'),M,H-121,width=135,height=74,mask='auto',preserveAspectRatio=True)
        c.drawImage(str(ROOT/'logos/sid_bigdata.png'),W-M-225,H-126,width=225,height=96,mask='auto',preserveAspectRatio=True)
        c.setFillColor(INK);c.setFont('Sans-B',10);c.drawString(M,H-193,'PROJET DONNÉES MULTIMÉDIA - M2 SID 2026-2027')
        c.setFont('Sans-B',69);c.drawString(M,H-285,'EmoVid')
        p=Paragraph("Reconnaissance d’émotions<br/>dans des extraits de film",sty('covertitle',fontName='Texte',fontSize=28,leading=35))
        _,h=p.wrap(CONTENT,150);p.drawOn(c,M,H-312-h)
        c.setFont('Sans',14);c.drawString(M,H-437,'Texte, audio, vidéo et fusion des modalités')
        yy=H-493;xs=[M+52,M+182,M+312];labs=['AUDIO','TEXTE','VIDÉO'];mods=['audio','texte','video']
        for x,label,mod in zip(xs,labs,mods):
            c.setStrokeColor(colors.HexColor(PAL[mod]));c.setLineWidth(2.3);c.line(x,yy-17,W-M-43,yy-92)
            c.setFillColor(colors.HexColor(PAL[mod]));c.circle(x,yy,5,fill=1,stroke=0)
            c.setFont('Sans-B',10);c.drawCentredString(x,yy+16,label)
        c.setFillColor(colors.HexColor(PAL['fusion']));c.circle(W-M-43,yy-92,8,fill=1,stroke=0)
        c.setFont('Sans-B',10);c.drawRightString(W-M,yy-119,'FUSION')
        url='https://gitlab.com/Maouloud3-0/projet-multimedia-m2sid'
        c.setFillColor(INK);c.setFont('Sans-B',10);c.drawString(M,196,'DÉPÔT DU PROJET')
        c.setFillColor(colors.HexColor(PAL['fusion']));c.setFont('Sans',11);c.drawString(M,180,url)
        c.linkURL(url,(M,176,M+c.stringWidth(url,'Sans',11),192),relative=1)
        c.setStrokeColor(colors.HexColor('#C9CFCE'));c.setLineWidth(.6);c.line(M,168,W-M,168)
        c.setFillColor(INK);c.setFont('Sans-B',12);c.drawString(M,143,'Groupe 4')
        c.setFont('Sans',12);c.drawString(M,122,'Gaoussou Maouloud KONTA');c.drawString(M,104,'Hanchen LI')
        c.setFont('Sans-B',11);c.drawString(M,67,'M2 SID — Université Toulouse III')
        c.setFont('Sans',10);c.drawString(M,49,'Projet Données multimédia, novembre 2026')
        c.restoreState()

class Report(BaseDocTemplate):
    def __init__(self):
        super().__init__(str(ROOT/'RAPPORT_v2.pdf'),pagesize=A4,leftMargin=M,rightMargin=M,topMargin=57,bottomMargin=50,
            title="EmoVid - Reconnaissance d'émotions dans des extraits de film",author='Gaoussou Maouloud KONTA ; Hanchen LI',
            subject='Projet Données multimédia - M2 SID 2026-2027',creator='Atelier de rapport - Groupe 4')
        frame=Frame(M,50,CONTENT,H-107,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,id='body')
        self.addPageTemplates([PageTemplate('normal',[frame],onPageEnd=self.decor)])
        self.chapter='EmoVid';self.mod='commun';self.placements=[]
    def beforeDocument(self):self.chapter='EmoVid';self.mod='commun';self.placements=[]
    def decor(self,c,doc):
        if doc.page==1:return
        c.saveState();color=colors.HexColor(PAL[self.mod]);c.setFillColor(color);c.setFont('Sans-B',8.2)
        c.drawString(M,H-31,'EMOVID / M2 SID')
        c.setFont('Sans',8.2);c.drawRightString(W-M,H-31,self.chapter[:72])
        c.setStrokeColor(colors.HexColor('#D5DDE2'));c.setLineWidth(.5);c.line(M,36,W-M,36)
        c.setFont('Sans',8);c.setFillColor(MUTED);c.drawString(M,23,'Groupe 4 · Gaoussou Maouloud KONTA · Hanchen LI')
        c.setFillColor(color);c.setFont('Sans-B',11);c.drawRightString(W-M,22,f'{doc.page:02}')
        c.restoreState()
    def afterFlowable(self,f):
        if isinstance(f,Chapter):
            self.chapter=f.chapterTitle;self.mod=f.mod
            key='chap_'+str(self.seq.nextf('chapter'))
            self.canv.bookmarkPage(key);self.canv.addOutlineEntry(f.chapterTitle,key,level=0)
            self.notify('TOCEntry',(0,f.chapterTitle,self.page,key))
        elif isinstance(f,Paragraph) and f.style.name=='section':
            key='sec_'+str(self.seq.nextf('section'))
            self.canv.bookmarkPage(key);self.canv.addOutlineEntry(f.getPlainText(),key,level=1,closed=True)
        if not isinstance(f,PageBreak) and self.frame is not None:
            self.placements.append({'page':self.page,'type':type(f).__name__,'bottom':round(self.frame._y,2),
                  'text':f.getPlainText()[:90] if isinstance(f,Paragraph) else ''})

def widths(rows,total):
    clean=lambda s:re.sub('[*`]','',s)
    n=len(rows[0]); mins=[]; preferred=[]
    for j in range(n):
        texts=[clean(r[j]) for r in rows]
        longest=max(pdfmetrics.stringWidth(t,'Sans',9) for t in texts)
        minimum=max(pdfmetrics.stringWidth(word,'Sans',9) for t in texts for word in (t.split() or ['']))
        mins.append(min(minimum+14,total*.32));preferred.append(min(longest+16,total*.64))
    if sum(preferred)<=total:return [x*total/sum(preferred) for x in preferred]
    rest=total-sum(mins);flex=[max(0,p-m) for p,m in zip(preferred,mins)]
    if rest<0:return [x*total/sum(mins) for x in mins]
    return [m+rest*f/sum(flex) for m,f in zip(mins,flex)]

COUNTS={'fig':0,'table':0}
def table(lines,caption,mod):
    raw=[[x.strip() for x in line.strip().strip('|').split('|')] for line in lines]
    header,align,rows=raw[0],raw[1],raw[2:];n=len(header)
    numeric=[a.endswith(':') for a in align]
    allrows=[header]+rows
    styles=[];data=[]
    for i,row in enumerate(allrows):
        data.append([Paragraph(inline(t),sty('th' if i==0 else 'td',fontName='Sans-B' if i==0 else 'Sans',
                  fontSize=8.8,leading=11.2,spaceAfter=0,textColor=colors.white if i==0 else INK,
                  alignment=TA_RIGHT if numeric[j] else TA_LEFT)) for j,t in enumerate(row)])
    t=Table(data,colWidths=widths(allrows,CONTENT),repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(PAL[mod])),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor(PALE[mod])]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor(PAL[mod]))]))
    COUNTS['table']+=1
    cap=Paragraph(f"<b>Tableau {COUNTS['table']:02}</b> · "+inline(caption or ''),sty('tablecaption',fontName='Sans',fontSize=9,leading=11.6,textColor=MUTED,spaceBefore=7,spaceAfter=5,keepWithNext=1))
    return [cap,t,Spacer(1,9)]

def figure(path,caption,size,mod):
    COUNTS['fig']+=1
    p=ROOT/path;w,h=ImageReader(str(p)).getSize()
    width=CONTENT if 'schema_' in path or 'confusion_' in path else min(CONTENT,float(size or 16)*28.346)
    if 'confusion_' in path and not 'fusion' in path.replace('confusion_',''):
        width=CONTENT*.94
    height=width*h/w
    pic=Image(str(p),width=width,height=height,hAlign='CENTER')
    cap=Paragraph(f"<b>Figure {COUNTS['fig']:02}</b> · "+inline(caption),CAPTION)
    return [KeepTogether([Spacer(1,6),pic,Spacer(1,5),cap])]

def convert(text):
    lines=text.splitlines();story=[];i=0;mod='commun'
    while i<len(lines):
        l=lines[i].strip()
        if not l or l=='---':i+=1;continue
        if l.startswith('## '):
            title=l[3:]
            n=title.split('.')[0]
            mod={'5':'audio','6':'texte','7':'video','8':'fusion','9':'fusion'}.get(n,'commun')
            if story:story += [Spacer(1,15),CondPageBreak(270)]
            story.append(Chapter(title,mod));i+=1
        elif l.startswith('### '):
            story.append(CondPageBreak(100))
            story.append(Paragraph(inline(l[4:]),sty('section',fontName='Sans-B',fontSize=14,leading=17,textColor=colors.HexColor(PAL[mod]),spaceBefore=11,spaceAfter=7,keepWithNext=1)));i+=1
        elif l.startswith('#### '):
            story.append(Paragraph(inline(l[5:]),sty('subsection',fontName='Sans-B',fontSize=11.5,leading=14,spaceBefore=9,spaceAfter=5,keepWithNext=1)));i+=1
        elif l.startswith('> **'):
            name=l[4:-2];parts=[];i+=1
            while i<len(lines) and lines[i].startswith('>'):
                parts.append(lines[i][1:].strip());i+=1
            story.append(Banner(name,' '.join(parts),mod))
        elif l.startswith('!['):
            m=re.match(r'!\[(.*)\]\(([^ )]+)(?: "([\d.]+)")?\)',l)
            story+=figure(m[2],m[1],m[3],mod);i+=1
        elif l.startswith('Tableau :') or l.startswith('|'):
            cap=l[len('Tableau :'):].strip() if l.startswith('Tableau :') else ''
            if cap:i+=1
            j=i
            while j<len(lines) and lines[j].startswith('|'):j+=1
            story+=table(lines[i:j],cap,mod);i=j
        elif l.startswith('```'):
            j=i+1
            while j<len(lines) and not lines[j].startswith('```'):j+=1
            paras=[Paragraph(inline(line),sty('formula',fontName='Sans',fontSize=9,leading=13,spaceAfter=3)) for line in lines[i+1:j]]
            t=Table([[p] for p in paras],colWidths=[CONTENT]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor(PALE[mod])),('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,0),9),('BOTTOMPADDING',(0,-1),(-1,-1),9)]))
            story += [KeepTogether([t]),Spacer(1,8)];i=j+1
        elif re.match(r'^(- |\d+\. )',l):
            number=re.match(r'^(\d+)\. ',l);bullet=number[1]+'.' if number else '•'
            first=re.sub(r'^(- |\d+\. )','',l);i+=1
            while i<len(lines) and lines[i].startswith('  ') and not lines[i].lstrip().startswith('- '):
                first+=' '+lines[i].strip();i+=1
            story.append(Paragraph(inline(first),sty('bullet',leftIndent=13,firstLineIndent=0,bulletIndent=1,bulletFontName='Sans',bulletFontSize=9,spaceAfter=4),bulletText=bullet))
            while i<len(lines) and lines[i].startswith('  '):
                sub=lines[i].strip();i+=1
                while i<len(lines) and lines[i].startswith('    '):sub+=' '+lines[i].strip();i+=1
                story.append(Paragraph(inline(sub.lstrip('- ')),sty('subbullet',leftIndent=24,bulletIndent=13,bulletFontName='Sans',spaceAfter=4),bulletText='–'))
        else:
            j=i+1
            while j<len(lines) and lines[j].strip() and not re.match(r'^(#|!\[|Tableau :|\||```|>|- |\d+\. )',lines[j]):j+=1
            story.append(Paragraph(inline(' '.join(x.strip() for x in lines[i:j])),BODY));i=j
    # Solidariser les titres avec les figures ; les tableaux restent divisibles.
    grouped=[];i=0
    def heading(f):
        return isinstance(f,Paragraph) and f.style.name in ('chapter','section','subsection')
    while i<len(story):
        if not heading(story[i]):grouped.append(story[i]);i+=1;continue
        group=[story[i]];i+=1
        while i<len(story) and (heading(story[i]) or isinstance(story[i],CondPageBreak)):
            if heading(story[i]):group.append(story[i])
            i+=1
        if i<len(story) and isinstance(story[i],KeepTogether):
            group+=story[i]._content;i+=1;grouped.append(KeepTogether(group))
        else:grouped+=group
    return grouped

def front():
    toc=TableOfContents(dotsMinLevel=0)
    toc.levelStyles=[sty('toc',fontName='Sans',fontSize=11,leading=18.5,spaceBefore=4,spaceAfter=2)]
    contents=[Cover(),PageBreak(),Paragraph('Lire le rapport',sty('fronttitle',fontName='Sans-B',fontSize=27,leading=32,spaceAfter=10)),
       Paragraph('Une démarche commune, trois modalités, une décision finale.',sty('introline',fontName='Texte-I',fontSize=12,leading=17,textColor=MUTED,spaceAfter=20)),
       toc,Spacer(1,20),Banner('Repères de lecture','Les encadrés « En bref » annoncent la question de chaque chapitre. Les encadrés « À retenir » en donnent le résultat. Les scores cv servent au choix des modèles ; dev et test désignent les ensembles réservés à leur évaluation.','commun'),PageBreak(),
       Paragraph("Vue d’ensemble",sty('fronttitle',fontName='Sans-B',fontSize=27,leading=32,spaceAfter=10)),
       Paragraph('Du corpus à une prédiction parmi huit émotions',sty('introline',fontName='Texte-I',fontSize=13,leading=18,textColor=MUTED,spaceAfter=13))]
    contents+=figure('figures/vue_ensemble.png',"Chaîne complète. Les scores affichés sont les F1 macro de validation croisée (cv) et de développement (dev). La transcription n'a pas de score d'erreur indépendant ; ses résultats sont des effectifs.",17,'commun')
    contents+=[Banner('Le principe de comparaison','N1 : mesures simples. N2 : encodeurs déjà entraînés, utilisés comme extracteurs de vecteurs. N3 : apprentissage supplémentaire. Pour chaque modalité, les meilleurs modèles sont combinés avant la fusion.','commun'),
       Paragraph("<b>Un protocole unique.</b> La validation croisée exclut des films entiers de chaque apprentissage. Le jeu de développement ne sert qu'à mesurer ; le jeu de test est évalué à la fin du point 5.",BODY),PageBreak()]
    return contents

def control():
    import pypdfium2 as pdfium
    from pypdf import PdfReader
    from PIL import Image as PILImage, ImageDraw
    out=ROOT/'controle/pages';out.mkdir(parents=True,exist_ok=True)
    pdf=pdfium.PdfDocument(str(ROOT/'RAPPORT_v2.pdf'));reader=PdfReader(ROOT/'RAPPORT_v2.pdf')
    for old in out.iterdir():
        match=re.fullmatch(r'(page|planche)_(\d+)\.(png|jpg)',old.name)
        if match and int(match[2]) > (len(pdf) if match[1]=='page' else (len(pdf)+3)//4):
            old.unlink()
    for i,page in enumerate(pdf):page.render(scale=1.4).to_pil().save(out/f'page_{i+1:02}.png')
    for start in range(0,len(pdf),4):
        sheet=PILImage.new('RGB',(1240,1780),'#D8DDE1');draw=ImageDraw.Draw(sheet)
        for k in range(start,min(start+4,len(pdf))):
            im=PILImage.open(out/f'page_{k+1:02}.png');im.thumbnail((600,849))
            x=(k-start)%2*620+10;y=(k-start)//2*890+25;sheet.paste(im,(x,y));draw.text((x,y-19),str(k+1),fill='black')
        sheet.save(out/f'planche_{start//4+1:02}.jpg')
    text='\n\n'.join(f'PAGE {i+1}\n'+page.extract_text() for i,page in enumerate(reader.pages))
    (ROOT/'controle/texte_pdf.txt').write_text(text,encoding='utf-8')
    print(len(pdf),'pages rendues pour contrôle visuel')

def build():
    (ROOT/'controle').mkdir(exist_ok=True)
    COUNTS.update(fig=0,table=0)
    doc=Report();story=front()+convert((ROOT/'RAPPORT_v2.md').read_text(encoding='utf-8'))
    doc.multiBuild(story,maxPasses=10)
    (ROOT/'controle/mise_en_page.json').write_text(json.dumps(doc.placements,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Rapport généré :',COUNTS)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--figures',action='store_true');parser.add_argument('--controle',action='store_true');a=parser.parse_args()
    if a.figures:
        subprocess.run([sys.executable,str(ROOT/'dessiner_figures.py')],check=True)
    build()
    if a.controle:control()
