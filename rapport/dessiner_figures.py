"""Redessine les figures depuis les résultats copiés ; ne réentraîne aucun modèle."""
from pathlib import Path
import os, sys, json, textwrap
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT/'outils'))
os.environ['MPLCONFIGDIR'] = str(ROOT/'controle'/'matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
from sklearn.metrics import f1_score, confusion_matrix

DATA, FIG = ROOT/'donnees', ROOT/'figures'
FIG.mkdir(exist_ok=True)
for f in (ROOT/'polices').glob('calibri*.ttf'):
    font_manager.fontManager.addfont(str(f))
INK='#213442'; MUTED='#63717B'; GRID='#DCE3E7'
COL={'audio':'#A36224','texte':'#16766D','video':'#365F94','fusion':'#795071','commun':'#365363'}
LIGHT={'audio':'#F8F0E5','texte':'#EAF5F1','video':'#EDF2F9','fusion':'#F4EDF3','commun':'#F1F4F6'}
plt.rcParams.update({'font.family':'Calibri','font.size':10,'text.color':INK,'axes.labelcolor':INK,
 'xtick.color':MUTED,'ytick.color':MUTED,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':GRID,'axes.titleweight':'bold','axes.titlesize':12,'axes.labelsize':10,
 'xtick.labelsize':9,'ytick.labelsize':9,'legend.frameon':False,'figure.facecolor':'white',
 'savefig.facecolor':'white','pdf.fonttype':42})
EMOTIONS=['amusement','anger','awe','contentment','disgust','excitement','fear','sadness']
SHORT=['amu','ang','awe','con','dis','exc','fea','sad']
corpus=pd.read_csv(DATA/'corpus.csv',dtype={'film_id':str},keep_default_na=False)
y=corpus.emotion.to_numpy(str)
masks={p:(corpus.partition==p).to_numpy() for p in ['train','dev','test']}

def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=240,bbox_inches='tight',pad_inches=.08)
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.08,metadata={'Creator':'Atelier de rapport','Author':'Groupe 4 - M2 SID'})
    plt.close(fig)

def load(folder,name):
    a=np.load(DATA/folder/f'probas_{name}.npz')
    assert np.array_equal(a['video_id'].astype(str),corpus.video_id.to_numpy(str))
    assert list(a['emotions'].astype(str))==EMOTIONS
    return a['probabilites'],a['probabilites_hors_pli']

def metric(p,part):
    m=masks[part]
    return f1_score(y[m],np.array(EMOTIONS)[p[m].argmax(1)],average='macro')

def matrix(ax,p,part,title):
    m=masks[part]
    cm=confusion_matrix(y[m],np.array(EMOTIONS)[p[m].argmax(1)],labels=EMOTIONS)
    pct=cm/cm.sum(1,keepdims=True)
    cmap=LinearSegmentedColormap.from_list('commun',['#F5F7FA','#9DB7CD','#183F65'])
    im=ax.imshow(pct,cmap=cmap,vmin=0,vmax=1)
    for i in range(8):
        for j in range(8):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=9,color='white' if pct[i,j]>.53 else INK)
    ax.set_xticks(range(8),SHORT);ax.set_yticks(range(8),SHORT)
    ax.tick_params(length=0,pad=4)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title(title+'\nF1 macro '+f'{metric(p,part):.3f}'.replace('.',','),pad=9,fontsize=11)
    return im

MODS={
 'audio':('point4_audio',[('n1','N1 · 221 descripteurs'),('emb_whisper','N2 · Whisper, couche 16'),('att_whisper','N3 · Whisper + attention'),('audio','Final · 4 modèles')]),
 'texte':('point4_texte',[('tfidf','N1 · TF-IDF'),('emb_e5','N2 · e5 gelé'),('ft_ekman','N3 · ekman affiné'),('texte','Final · 6 modèles')]),
 'video':('point4_video',[('n1','N1 · 270 descripteurs'),('emb_pe','N2 · PE + visages'),('att_pe','N3 · PE + attention'),('video','Final · 6 modèles')])}

def charts():
    audit={}
    for mod,(folder,models) in MODS.items():
        for phase,part in [('cv','train'),('dev','dev')]:
            fig,axes=plt.subplots(2,2,figsize=(7.0,7.05))
            for ax,(name,title) in zip(axes.flat,models):
                p,h=load(folder,name); values=h if part=='train' else p
                im=matrix(ax,values,part,title)
                audit[f'{mod}/{name}/{phase}']=metric(values,part)
            fig.subplots_adjust(left=.06,right=.94,bottom=.07,top=.94,wspace=.27,hspace=.37)
            bar=fig.add_axes([.20,.008,.60,.012]); cb=fig.colorbar(im,cax=bar,orientation='horizontal',ticks=[0,.25,.5,.75,1])
            cb.ax.set_xticklabels(['0 %','25 %','50 %','75 %','100 %']);cb.ax.tick_params(length=0,labelsize=8)
            save(fig,f'confusion_{mod}_{phase}')
            for pair in range(2):
                fig,axes=plt.subplots(1,2,figsize=(7,3.6))
                for ax,(name,title) in zip(axes,models[pair*2:pair*2+2]):
                    p,h=load(folder,name)
                    im=matrix(ax,h if part=='train' else p,part,title)
                fig.subplots_adjust(left=.06,right=.94,bottom=.14,top=.88,wspace=.27)
                bar=fig.add_axes([.23,.025,.54,.018]);cb=fig.colorbar(im,cax=bar,orientation='horizontal',ticks=[0,.5,1])
                cb.ax.set_xticklabels(['0 %','50 %','100 %']);cb.ax.tick_params(length=0,labelsize=8)
                save(fig,f'confusion_{mod}_{phase}_{pair+1}')
    pf,hf=load('point5_fusion','finale');pb,_=load('point5_fusion','precoce_brut');pc,_=load('point5_fusion','attention')
    fig,axes=plt.subplots(2,3,figsize=(10.5,7.2))
    for ax,p,part,title in zip(axes.flat,[hf,pf,pf,pb,pc],['train','dev','test','dev','dev'],['A · CV','A · dev','A · test','B · dev','C · dev']):matrix(ax,p,part,title)
    axes[1,2].axis('off');axes[1,2].text(0,.85,'LECTURE COMMUNE',weight='bold',fontsize=13)
    axes[1,2].text(0,.65,'Lignes : émotion réelle\nColonnes : prédiction\nCases : effectifs\nCouleur : part par ligne\n\nÉchelle identique : 0 à 100 %',fontsize=12,va='top',linespacing=1.6)
    fig.tight_layout(h_pad=2.5,w_pad=2);save(fig,'confusion_fusion')

    fig,(a,b)=plt.subplots(1,2,figsize=(7,2.7),gridspec_kw={'width_ratios':[1.4,1]})
    counts=corpus.emotion.value_counts(normalize=True).reindex(EMOTIONS)*100
    a.bar(SHORT,counts,color=COL['commun'],width=.65)
    for i,v in enumerate(counts):a.text(i,v+.7,f'{v:.1f}'.replace('.',','),ha='center',fontsize=8)
    a.axhline(12.5,color=MUTED,ls='--',lw=.8);a.set_ylim(0,27);a.set_ylabel('Part des extraits (%)')
    b.hist(corpus.duree.astype(float),bins=np.arange(4,31),color=COL['commun'],rwidth=.88)
    b.set_xlabel('Durée (secondes)');b.set_ylabel('Extraits');fig.tight_layout();save(fig,'corpus')

    fig,ax=plt.subplots(figsize=(7,3.15))
    for mod,file in [('audio','scores_fusion_audio.csv'),('texte','scores_fusion_n3.csv'),('video','scores_fusion_video.csv')]:
        d=pd.read_csv(DATA/('point4_'+mod)/file);best=d.loc[d.f1_cv.idxmax()]
        ax.plot(d.k,d.f1_cv,'o-',label=mod,color=COL[mod],ms=4,lw=1.8)
        ax.plot(best.k,best.f1_cv,'o',mfc='white',mec=COL[mod],mew=2,ms=9)
        ax.annotate(f'k = {int(best.k)}',(best.k,best.f1_cv),xytext=(9,9),textcoords='offset points',fontsize=9,color=COL[mod])
    ax.set_xlabel('Nombre k de modèles moyennés');ax.set_ylabel('F1 cv');ax.legend(loc='center right');ax.grid(axis='y',color=GRID,lw=.5);fig.tight_layout();save(fig,'choix_k')

    d=pd.read_csv(DATA/'point4_audio/scores_n2.csv',dtype={'couche':str});d=d[(d.etape=='A')&(d.regroupement=='moyenne')&(d.couche!='moyenne')]
    fig,ax=plt.subplots(figsize=(7,3.1))
    for enc,label,c,ls in [('whisper','Whisper · 32 couches',COL['audio'],'-'),('wavlm','WavLM · 24','#4B6575','--'),('w2v2_emo','wav2vec2 · 12','#947550','-.'),('ast','AST · 12','#A5A8AA',':')]:
        a=d[d.encodeur==enc].copy();a['depth']=a.couche.astype(int);a=a.sort_values('depth')
        ax.plot(a.depth/a.depth.max(),a.f1_cv,marker='o',ls=ls,color=c,label=label,ms=4)
    ax.set_ylabel('F1 cv moyen par pli');ax.set_xlabel('Profondeur relative : entrée 0 → dernière couche 1');ax.legend(fontsize=9,ncol=2,loc='lower right');ax.grid(axis='y',color=GRID,lw=.5);fig.tight_layout();save(fig,'couches_audio')

    for mod,file,name in [('audio','importance_familles_n1.csv','importance_familles_n1'),('video','importance_familles_n1_video.csv','importance_familles_n1_video')]:
        d=pd.read_csv(DATA/file).sort_values('chute_f1')
        labels=[relabel(t)+f' [{int(n)}]' for t,n in zip(d.famille,d.nb_descripteurs)]
        fig,ax=plt.subplots(figsize=(7,4.6 if mod=='audio' else 2.9))
        ax.barh(np.arange(len(d)),d.chute_f1,xerr=d.ecart_type,color=COL[mod],height=.62,error_kw={'ecolor':'#99A6B1','capsize':2,'elinewidth':.9})
        ax.set_yticks(np.arange(len(d)),labels,fontsize=9);ax.axvline(0,color=MUTED,lw=.7)
        ax.set_xlabel('Baisse de F1 macro après permutation');ax.grid(axis='x',color=GRID,lw=.5);ax.set_axisbelow(True)
        fig.tight_layout();save(fig,name)

    _,hv=load('point4_video','video');_,ht=load('point4_texte','texte');_,ha=load('point4_audio','audio')
    grid=np.round(np.arange(0,1.01,.1),1)
    vals=np.array([[metric(.3*hv+wt*ht+wa*ha,'train') for wt in grid] for wa in grid])
    fig,ax=plt.subplots(figsize=(5.3,3.6))
    cmap=LinearSegmentedColormap.from_list('fusion',['#F4EDF3','#BFA5B9','#68425F'])
    im=ax.imshow(vals,origin='lower',cmap=cmap)
    ax.set_xticks(range(11),[str(x).replace('.',',') for x in grid],fontsize=8);ax.set_yticks(range(11),[str(x).replace('.',',') for x in grid],fontsize=8)
    ax.set_xlabel('Poids du texte');ax.set_ylabel("Poids de l'audio");ax.set_title('Poids vidéo fixé à 0,3',fontsize=11)
    ax.plot(1,1,'s',mfc='none',mec='white',mew=1.8,ms=13);ax.plot(0,0,'o',mfc='none',mec='#F3C96E',mew=1.5,ms=9)
    fig.colorbar(im,ax=ax,label='F1 cv');fig.tight_layout();save(fig,'poids_fusion')

    fig,ax=plt.subplots(figsize=(7,3.3))
    for i,(mod,p) in enumerate([('audio',load('point4_audio','audio')[0]),('texte',load('point4_texte','texte')[0]),('video',load('point4_video','video')[0]),('fusion',pf)]):
        scores=f1_score(y[masks['dev']],np.array(EMOTIONS)[p[masks['dev']].argmax(1)],labels=EMOTIONS,average=None)
        ax.bar(np.arange(8)+(i-1.5)*.2,scores,.18,color=COL[mod],label=mod)
    ax.set_xticks(range(8),SHORT);ax.set_ylim(0,.7);ax.set_ylabel('F1 par émotion · dev');ax.legend(ncol=4,loc='upper center');ax.grid(axis='y',color=GRID,lw=.5);ax.set_axisbelow(True);fig.tight_layout();save(fig,'f1_par_emotion')
    comparison=pd.read_csv(DATA/'point5_fusion/scores_comparaison.csv')
    expected={'seule_audio':(.271,.270,.297),'seule_texte':(.362,.387,.365),'seule_video':(.450,.461,.413),'tardive_ponderee':(.455,.486,.419),'precoce_brut':(.428,.438,None),'attention':(.439,.450,None)}
    for name,want in expected.items():
        row=comparison[comparison.systeme==name].iloc[0]
        for col,val in zip(['f1_cv','f1_dev','f1_test'],want):
            if val is not None:assert round(row[col],3)==val,(name,col,row[col],val)
    params=sum([1801*256+256,1280*256+256,1025*256+256,3*256,257,256*8+8])
    audit['parametres_option_C']=params
    audit['scores_finaux']=comparison[comparison.systeme.isin(expected)].fillna('').to_dict('records')
    for mod in ['audio','texte','video']:
        p,h=load('point4_'+mod,mod);row=comparison[comparison.systeme=='seule_'+mod].iloc[0]
        for part,col in [('train','f1_cv'),('dev','f1_dev'),('test','f1_test')]:
            got=metric(h if part=='train' else p,part);assert abs(got-row[col])<1e-12
    for part,col in [('train','f1_cv'),('dev','f1_dev'),('test','f1_test')]:
        got=metric(hf if part=='train' else pf,part)
        assert abs(got-comparison[comparison.systeme=='tardive_ponderee'].iloc[0][col])<1e-12
    (ROOT/'controle/verification_chiffres.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')

def relabel(t):
    return (t.replace(' (librosa)','').replace(' (eGeMAPS)',' · eGeMAPS')
     .replace('qualité de voix (jitter, shimmer, HNR, H1-H2, H1-A3)','Qualité de voix')
     .replace('centroïde, largeur, rolloff, platitude','Forme spectrale · librosa')
     .replace('énergie, passages par zéro, attaques','Énergie, zéro, attaques')
     .replace('rythme : segments voisés et pauses','Segments voisés et pauses')
     .replace('chroma : notes de musique','Notes de musique'))

def canvas(h=3.1):
    fig,ax=plt.subplots(figsize=(7.1,h));ax.set_xlim(0,100);ax.set_ylim(0,100);ax.axis('off');return fig,ax

def box(ax,x,y,w,h,text,mod='commun',fill=True,fs=10,bold=False):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.5,rounding_size=1.5',linewidth=1.1,edgecolor=COL[mod],facecolor=LIGHT[mod] if fill else 'white')
    ax.add_patch(patch);ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=INK,linespacing=1.35,weight='bold' if bold else 'normal')

def arrow(ax,x1,y1,x2,y2,c=INK):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=10,lw=1,color=c,shrinkA=2,shrinkB=2))

def label(ax,x,y,text,fs=9,color=MUTED,weight='normal',ha='center'):
    ax.text(x,y,text,ha=ha,va='center',fontsize=fs,color=color,weight=weight,linespacing=1.35)

def flow(name,nodes,mod='commun',note='',h=2.25):
    fig,ax=canvas(h);n=len(nodes);gap=3;w=(98-(n-1)*gap)/n
    for i,t in enumerate(nodes):
        x=1+i*(w+gap);box(ax,x,30,w,50,t,mod,fs=10)
        if i<n-1:arrow(ax,x+w,55,x+w+gap,55)
    if note:label(ax,50,10,note,9)
    save(fig,name)

def diagrams():
    # Vue d'ensemble : scores finaux de CV et dev uniquement.
    fig,ax=canvas(5.0)
    box(ax,5,82,90,15,'CORPUS · 3 088 extraits / 350 films\nPartition par film : 250 train · 50 dev · 50 test','commun',fs=11,bold=True)
    box(ax,5,61,90,14,'TRANSCRIPTION · Whisper + Qwen + contrôle acoustique\n1 032 extraits relus → 2 025 dialogues anglais','commun',fs=11)
    arrow(ax,50,82,50,75)
    for x,mod,txt in [(1,'audio','AUDIO\nSon · 221 descripteurs\nEncodeurs + attention\n4 modèles\nF1 cv 0,271 · dev 0,270'),(35,'texte','TEXTE\nDescription + dialogue\nTF-IDF, encodeurs, affinage\n6 modèles\nF1 cv 0,362 · dev 0,387'),(69,'video','VIDÉO\n16 images + visages\nEncodeurs + attention\n6 modèles\nF1 cv 0,450 · dev 0,461')]:
        box(ax,x,28,30,25,txt,mod,fs=10);arrow(ax,x+15,61,x+15,53)
        arrow(ax,x+15,28,50,19)
    box(ax,9,2,82,17,'FUSION A · moyenne des probabilités\nVidéo 0,6 + texte 0,2 + audio 0,2\nF1 cv 0,455 · dev 0,486','fusion',fs=11,bold=True)
    save(fig,'vue_ensemble')

    fig,ax=canvas(3.5)
    box(ax,1,35,20,30,'WAV mono\n16 kHz',fs=11)
    for yy,txt in [(72,'Whisper large-v3\ntranscription 1'),(40,'Qwen3-ASR-1.7B\ntranscription 2'),(8,'Silero + YAMNet\nparole, chant, musique')]:
        box(ax,29,yy,31,22,txt,fs=10);arrow(ax,21,50,29,yy+11)
    box(ax,70,43,29,32,'AIGUILLAGE\naccord + acoustique\ncas douteux : écoute',fs=11,bold=True)
    for yy in [83,51,19]:arrow(ax,60,yy,70,59)
    label(ax,83,22,'BandIt + Qwen sur voix\nsi séparation nécessaire',10)
    save(fig,'schema_transcription')

    fig,ax=canvas(4.1)
    box(ax,2,75,96,22,'ACCORD ENTRE DEUX SYSTÈMES INDÉPENDANTS\nSimilarité ≥ 0,90 · longueur courte / longue ≥ 0,85 · même langue\nMêmes négations et nombres · accord exact si ≤ 4 mots',fs=10,bold=True)
    box(ax,2,41,45,24,'PAROLE CONFIRMÉE\nSilero ≥ 20 % OU YAMNet ≥ 0,5\nET pic de chant < 0,5',fs=10)
    box(ax,53,41,45,24,'PAS DE DIALOGUE\nSorties vides + peu de parole\nSilero ≤ 5 % ET YAMNet < 0,3',fs=10)
    arrow(ax,25,75,25,65);arrow(ax,75,75,75,65)
    box(ax,2,4,45,22,'Accord + parole → texte accepté\nSinon → écoute humaine',fs=10,bold=True)
    box(ax,53,4,45,22,'Voix vide + RMS < 0,01\navec peu de parole : autre voie\n3 sorties saines vides : priorité',fs=9.5)
    arrow(ax,25,41,25,26);arrow(ax,75,41,75,26);save(fig,'schema_routage')

    fig,ax=canvas(3.4)
    for j in range(5):label(ax,17+j*11,94,f'Pli {j+1}',10)
    for i in range(5):
        label(ax,1,79-i*16,f'{i+1}',10)
        for j in range(5):box(ax,12+j*11,73-i*16,10,12,'prédit' if i==j else 'apprend','texte' if i==j else 'commun',fs=8)
    box(ax,73,45,26,41,'2 207 prédictions\nhors pli\n\nChaque film est\nexclu une fois',fs=10,bold=True)
    arrow(ax,67,52,73,65);label(ax,83,17,'F1 cv → choix\ndes réglages',10);arrow(ax,86,45,86,27)
    save(fig,'schema_validation_croisee')

    fig,ax=canvas(3.25)
    for yy,title in [(70,'N1 · mesures explicites + classifieur'),(40,'N2 · encodeur gelé + classifieur'),(10,'N3 · attention ou encodeur affiné')]:
        box(ax,1,yy,56,22,title,fs=10);arrow(ax,57,yy+11,67,51)
    box(ax,67,31,32,40,'8 probabilités / modèle\nClassement par F1 cv\nMoyenne des k meilleurs\nk choisi en CV',fs=10,bold=True)
    save(fig,'schema_parcours_modalite')

    fig,ax=canvas(3)
    for i,t in enumerate(['Whisper + régression','Whisper + attention','WavLM + régression','WavLM + attention']):
        box(ax,1,75-i*23,40,18,t,'audio',fs=10);arrow(ax,41,84-i*23,52,50)
    box(ax,52,28,47,44,'MOYENNE ÉMOTION PAR ÉMOTION\np = (p₁ + p₂ + p₃ + p₄) / 4\n8 probabilités → émotion la plus probable','audio',fs=10,bold=True)
    save(fig,'schema_combinaison')
    flow('schema_n1_audio',['Son mono\nTrames courtes','eGeMAPS : 88\nlibrosa : 133','221 valeurs\nstandardisées','Régression\n→ 8 émotions'],'audio','Moyenne et écart-type appris sur les données d’apprentissage du pli.')
    flow('schema_n2_encodeur',['Son →\nlog-mel\n128 bandes','WHISPER GELÉ\n32 couches\nsorties enregistrées','Vecteurs de\nla couche choisie\n→ moyenne','Régression\napprise\n→ 8 émotions'],'audio',"Vecteurs produits par les couches, jamais les poids. Le décodeur n'est pas utilisé.",h=2.6)
    flow('schema_n3_attention',['16 tranches\n× 1 280\nWhisper gelé','Projection\n1 280 → 256\nGELU + dropout','16 notes\n→ softmax\n→ moyenne','256 valeurs\n→ 8 émotions'],'audio','Les notes apprises pondèrent les moments ; seules les trois couches de la tête sont entraînées.',h=2.6)
    flow('schema_tfidf',['Description\n+ dialogue','Mots et bigrammes\nTF × IDF','8 167 valeurs\nvocabulaire train','Régression\n→ 8 émotions'],'texte','Vocabulaire et IDF réappris dans chaque pli ; vecteurs creux, sans standardisation centrée.')
    flow('schema_n2_texte',['Texte\n→ jetons','ENCODEUR GELÉ\ne5, mpnet, bge…','Vecteurs des jetons\n→ moyenne\n768 / 1 024 valeurs','Régression\napprise\n→ 8 émotions'],'texte',"Extraction par encodeurs uniquement ; aucun décodeur n'est utilisé.",h=2.6)
    flow('schema_n3_texte',['Description\n[séparation]\ndialogue','ENCODEUR AFFINÉ\ntoutes les couches\nmises à jour','Premier jeton\n→ vecteur\nde la paire','Tête apprise\n→ 8 émotions'],'texte','4 époques · 256 jetons maximum · taux d’apprentissage choisi par validation croisée.',h=2.6)
    fig,ax=canvas(1.8)
    for i in range(16):
        ax.add_patch(Rectangle((2+i*5.3,40),5.3,25,fc=LIGHT['video'],ec=COL['video'],lw=.7))
        ax.plot(4.65+i*5.3,52.5,'o',color=COL['video'],ms=3)
    ax.add_patch(Rectangle((86.8,40),11,25,fc='#E6E7E7',ec='#AAB0B5'))
    label(ax,44,83,'16 images au milieu de 16 intervalles égaux',11,weight='bold')
    label(ax,44,18,'Partie utile de l’extrait',10);label(ax,92,16,'Écran de fin\nexclu',9)
    save(fig,'schema_images_video')
    flow('schema_n1_video',['16 images\n160 × 90\nespace HSV','6 couleurs / lumière\n1 mouvement\n128 histogrammes','Moyenne + écart-type\nsur 16 images\n270 valeurs','Régression\nC = 0,0003\n→ 8 émotions'],'video',h=2.6)
    flow('schema_n2_video',['16 images\n→ encodeur gelé','Moyenne\ndes vecteurs\n+ résumé des visages','Standardisation\napprise dans le pli','Régression\n→ 8 émotions'],'video',"Encodeurs d'images uniquement ; aucun décodeur n'est utilisé.",h=2.5)
    fig,ax=canvas(3.1)
    box(ax,1,35,25,35,'YuNet ≥ 0,6\n2 plus grands visages\nMarge de 20 %','video',fs=10)
    box(ax,36,65,37,27,'ViT / FER2013 gelé\n768 valeurs + 7 probabilités','video',fs=10)
    box(ax,36,10,37,34,'HSEmotion gelé\n1 280 valeurs + 8 probabilités\n+ valence et éveil','video',fs=10)
    arrow(ax,26,52,36,78);arrow(ax,26,52,36,27)
    box(ax,81,28,18,46,'Résumé\nsur\nl’extrait','video',fs=10)
    arrow(ax,73,78,81,51);arrow(ax,73,27,81,51);save(fig,'schema_visages')
    fig,ax=canvas(3.4)
    for x,head,txt in [(1,'A · PROBABILITÉS','3 × 8 probabilités\n↓\nMoyenne pondérée\n↓\n8 probabilités'),(35,'B · VECTEURS','5 413 valeurs\n↓\nRégression logistique\n↓\n8 probabilités'),(69,'C · MORCEAUX','16 images + 1 texte\n+ 16 tranches de son\n↓\nProjection + attention\n↓\n8 probabilités')]:
        label(ax,x+15,92,head,11,COL['fusion'],'bold');box(ax,x,7,30,72,txt,'fusion',fs=11)
    save(fig,'schema_fusions')

    fig,ax=canvas(6.0)
    for x,mod,head,enc,dim in [
       (1,'video','16 IMAGES','PE + visage ViT\n1 024 + 777 valeurs','16 × 1 801'),
       (35,'texte','1 TEXTE','e5-large-v2\n1 024 + indicateur dialogue','1 × 1 025'),
       (69,'audio','16 TRANCHES','WHISPER : ENCODEUR\nmoyenne des couches','16 × 1 280')]:
        label(ax,x+15,97,head,11,COL[mod],'bold');box(ax,x,77,30,15,enc,mod,fs=10)
        label(ax,x+15,73,dim,11,COL[mod],'bold');arrow(ax,x+15,69,x+15,63)
        box(ax,x,49,30,14,'Projection propre → 256\nGELU + dropout\n+ badge de modalité',mod,fs=10)
        arrow(ax,x+15,49,50,39)
    label(ax,50,105,"Encodeurs gelés · aucun décodeur utilisé",9)
    box(ax,7,23,86,16,'POINT DE RENCONTRE : UNE SEULE ATTENTION SUR LES 33 MORCEAUX\n33 notes → un softmax commun → 33 poids de somme 1\nMoyenne pondérée → un seul vecteur de 256 valeurs','fusion',fs=11,bold=True)
    arrow(ax,50,23,50,18)
    box(ax,22,6,56,12,'Dropout → couche 256 → 8 → softmax\n8 probabilités d’émotion','fusion',fs=11,bold=True)
    label(ax,50,0,'≈ 1,05 M de paramètres appris · 95 % d’attention vidéo sur dev · sans self-attention',9)
    save(fig,'schema_option_c')

if __name__=='__main__':
    charts();diagrams();print('Figures redessinées et chiffres vérifiés.')
