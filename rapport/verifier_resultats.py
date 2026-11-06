"""Audit reproductible des résultats conservés, sans nouvel entraînement."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parent
sys.path.append(str(ROOT/'outils'))
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,confusion_matrix
from pypdf import PdfReader

def main():
    data=ROOT/'donnees';df=pd.read_csv(data/'corpus.csv',dtype={'film_id':str},keep_default_na=False)
    classes=['amusement','anger','awe','contentment','disgust','excitement','fear','sadness']
    def load(folder,model):
        z=np.load(data/folder/f'probas_{model}.npz')
        assert np.array_equal(z['video_id'].astype(str),df.video_id.to_numpy(str))
        assert list(z['emotions'].astype(str))==classes
        assert np.array_equal(z['partition'].astype(str),df.partition.to_numpy(str))
        return z['probabilites'],z['probabilites_hors_pli']
    y=df.emotion.to_numpy(str);prediction=lambda p:np.array(classes)[p.argmax(1)]
    groups={p:set(df.loc[df.partition==p,'film_id']) for p in ['train','dev','test']}
    assert [len(groups[p]) for p in groups]==[250,50,50]
    assert not (groups['train']&groups['dev'] or groups['train']&groups['test'] or groups['dev']&groups['test'])
    assert [sum(df.partition==p) for p in groups]==[2207,442,439]
    assert len(df)==3088 and df.a_dialogue.sum()==2025
    systems={m:load('point4_'+m,m) for m in ['audio','texte','video']}
    systems.update({k:load('point5_fusion',n) for k,n in [('A','finale'),('B','precoce_brut'),('C','attention')]})
    expected={'audio':[.271,.270,.297],'texte':[.362,.387,.365],'video':[.450,.461,.413],
              'A':[.455,.486,.419],'B':[.428,.438,None],'C':[.439,.450,None]}
    records=[]
    for name,(p,h) in systems.items():
        row={'systeme':name}
        for part,want in zip(['train','dev','test'],expected[name]):
            if want is None:continue
            mask=(df.partition==part).to_numpy();a=h if part=='train' else p
            got=f1_score(y[mask],prediction(a[mask]),average='macro')
            assert round(got,3)==want,(name,part,got)
            row[part]=float(got)
        records.append(row)
    for index,part in [(0,'dev'),(1,'train')]:
        mask=(df.partition==part).to_numpy()
        mix=.6*systems['video'][index][mask]+.2*systems['texte'][index][mask]+.2*systems['audio'][index][mask]
        assert np.allclose(mix,systems['A'][index][mask],atol=1e-7)
    # Même tirage de films que le script de référence : graine 42, 2 000 tirages.
    def bootstrap(part):
        mask=(df.partition==part).to_numpy();films=df.loc[mask,'film_id'].to_numpy();truth=y[mask]
        pa=prediction(systems['A'][0][mask]);pv=prediction(systems['video'][0][mask])
        unique=np.unique(films);rng=np.random.default_rng(42)
        ma=np.array([confusion_matrix(truth[films==f],pa[films==f],labels=classes) for f in unique])
        mv=np.array([confusion_matrix(truth[films==f],pv[films==f],labels=classes) for f in unique])
        def f1(cm):
            den=cm.sum(0)+cm.sum(1)
            return np.divide(2*np.diag(cm),den,out=np.zeros(8),where=den!=0).mean()
        diffs=[]
        for _ in range(2000):
            idx=rng.choice(len(unique),len(unique));diffs.append(f1(ma[idx].sum(0))-f1(mv[idx].sum(0)))
        return np.percentile(diffs,[2.5,97.5]).tolist()
    intervals={part:bootstrap(part) for part in ['dev','test']}
    assert np.allclose(np.round(intervals['dev'],3),[.001,.053])
    assert np.allclose(np.round(intervals['test'],3),[-.017,.030])
    att=pd.read_csv(data/'point5_fusion/scores_attention.csv').iloc[0]
    parameters=json.loads(att.parametres)
    assert parameters['cachee']==256 and parameters['couches']==0 and parameters['epoques']==16
    assert round(att.attention_video*100)==95
    assert sum([1801*256+256,1280*256+256,1025*256+256,3*256,257,256*8+8])==1054985
    report=PdfReader(ROOT/'RAPPORT_v2.pdf')
    assert len(report.pages)>0 and all(len(p.extract_text().strip())>80 for p in report.pages)
    source=(ROOT/'RAPPORT_v2.md').read_text(encoding='utf-8')
    sections=source.split('## 8. Fusion')[0].split('## 1. Introduction')[1]
    assert not any(t in sections for t in ['F1 test','0,297 sur test','0,365 sur test','0,413 sur test'])
    assert source.count('> **En bref**')==10 and source.count('> **À retenir**')==10
    result={'extraits':len(df),'films':350,'dialogues_anglais':2025,'scores':records,'IC95_fusion_moins_video':intervals,
            'parametres_option_C':1054985,'attention_video':float(att.attention_video),'pages':len(report.pages),
            'metadonnees':dict(report.metadata),'verification':'Tous les contrôles ont réussi.'}
    (ROOT/'controle/audit_final.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Contrôles réussis : partition, 2 025 dialogues, 6 systèmes, poids, intervalles par films et architecture.')

if __name__=='__main__':main()
