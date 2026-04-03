#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import scanpy as sc
import pandas as pd
import seaborn as sns


# ### cd8t_b_mono_cell_embedding

# In[3]:


hvgdf = pd.read_csv('./data/zheng68k_train_var.csv',index_col=0)
selected_gene = hvgdf.highly_variable.values

cell_geneemb = np.load('./data/cd8t_b_mono_geneemb_01B-resolution_singlecell_gene_embedding_f1_resolution.npy')

geneemb_merge = cell_geneemb[:,selected_gene,:].mean(0)
geneemb_merge.shape


# In[5]:


gene_list_df = pd.read_csv('../OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
gene_list = list(gene_list_df['gene_name'])


# In[6]:


gene_adata=sc.AnnData(pd.DataFrame(geneemb_merge,index=np.array(gene_list)[selected_gene]))
sc.pp.neighbors(gene_adata,use_rep='X')
sc.tl.umap(gene_adata)


# In[7]:


sc.tl.leiden(gene_adata,resolution=5)


# In[8]:


sc.pl.umap(gene_adata,color='leiden')


# In[9]:


# The function is modified from https://github.com/bowang-lab/scGPT
import collections
def get_metagenes(gdata):
    metagenes = collections.defaultdict(list)
    for x, y in zip(gdata.obs["leiden"], gdata.obs.index):
        metagenes[x].append(y)
    return metagenes

metagenes = get_metagenes(gene_adata)

# Obtain the set of gene programs from clusters with #genes >= 5
mgs = dict()
for mg, genes in metagenes.items():
    if len(genes) > 4:
        mgs[mg] = genes


# In[102]:


import pickle
with open ('mgs.pkl','wb') as f:
    pickle.dump(mgs,f)


# In[10]:


len(mgs)


# In[11]:


# The function is modified from https://github.com/bowang-lab/scGPT

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
def score_metagenes(adata, metagenes):
    for p, genes in tqdm(metagenes.items()):
        try:
            sc.tl.score_genes(adata, score_name=str(p) + "_SCORE", gene_list=genes)
            scores = np.array(adata.obs[str(p) + "_SCORE"].tolist()).reshape(-1, 1)
            scaler = MinMaxScaler()
            scores = scaler.fit_transform(scores)
            scores = list(scores.reshape(1, -1))[0]
            adata.obs[str(p) + "_SCORE"] = scores
        except Exception as e:
            adata.obs[str(p) + "_SCORE"] = 0.0


# In[12]:


adata = sc.read_h5ad('./data/zheng_downsampled_cd8t_b_mono.h5ad')

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

score_metagenes(adata,mgs)


# In[13]:


scorelist = [x for x in adata.obs.columns if x.__contains__('SCORE')]
genescoreadata = sc.AnnData(adata.obs[scorelist])
genescoreadata.obs = adata.obs.iloc[:,:3].copy()
sc.tl.rank_genes_groups(genescoreadata,groupby='label')


# In[14]:


rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
sc.pl.rank_genes_groups_matrixplot(genescoreadata, n_genes=5, standard_scale='var', cmap='Blues',save='celltype_module')


# In[15]:


print(mgs['18'],'\n',mgs['28'],'\n',mgs['12'])


# In[16]:


geneemb_mergedf = pd.DataFrame(geneemb_merge,index=np.array(gene_list)[selected_gene])
genesub = mgs['28']
cd8tembdf = geneemb_mergedf.loc[genesub].copy()


# In[17]:


import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


# In[18]:


# The function is modified from https://github.com/bowang-lab/scGPT
G = nx.Graph()
similarities = cosine_similarity(cd8tembdf)
genes = list(cd8tembdf.index.values)
similarities[similarities>0.9999]=0
edges = []
nz = list(zip(*similarities.nonzero()))
for n in tqdm(nz):
    edges.append((genes[n[0]], genes[n[1]],{'weight':similarities[n[0],n[1]]}))
G.add_nodes_from(genes)
G.add_edges_from(edges)


# In[19]:


widths = nx.get_edge_attributes(G, 'weight')
weightvalue = np.array(list(widths.values()))
scaled_weightvalue = (weightvalue-weightvalue.min())/(weightvalue.max()-weightvalue.min())*3


# In[20]:


widsorted = sorted(widths.items(), key=lambda x: x[1], reverse=True)


# In[21]:


toppair = np.array(list(widths))[weightvalue.argsort()<3]


# In[22]:


pos = nx.spring_layout(G, k=0.4, iterations=15, seed=42)

nx.draw_networkx_edges(G, pos,
                       edgelist = widths.keys(),
                       edge_color=list(widths.values()),
                       width=scaled_weightvalue,
                       edge_cmap=mpl.colormaps['cool'],
                       alpha=1)

nx.draw_networkx_labels(G, pos, font_size=15, font_family="sans-serif")

# edge weight labels
edge_labels = {widsorted[i][0]: f'rank{i+1}' for i in range(5)}
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
plt.savefig('figures/T_genemodule.pdf',bbox_inches='tight')


# ## Use emb for pySCENIC

# In[23]:


cell_geneemb = np.load('./data/cd8t_b_mono_geneemb_01B-resolution_singlecell_gene_embedding_f1_resolution.npy')


# In[25]:


selected_gene = adata.X.sum(0)>0
geneemb_merge = cell_geneemb[:,selected_gene,:].mean(0)
geneemb_merge.shape


# In[26]:


TF = pd.read_csv('./data/allTFs_hg38.txt',header=None).values.T[0]
sgene = np.array(gene_list)[selected_gene]
fltTF = [x for x in TF if x in sgene]
len(fltTF)


# In[27]:


fltgeneembdf = pd.DataFrame(geneemb_merge,index=sgene)


# In[28]:


coexplist=[]
for tf in tqdm(fltTF):
    tmpsim = cosine_similarity(fltgeneembdf.loc[fltTF[0],:].values.reshape(1,-1),fltgeneembdf)
    tmpsim[tmpsim>0.9999]=0
    tmpsimdf = pd.DataFrame(tmpsim,columns=sgene,index=['simi']).T
    tmpsimdf = tmpsimdf.sort_values('simi',ascending=False)
    for idx in range(1000):
        coexplist.append([tf,tmpsimdf.index[idx],tmpsimdf.iloc[idx,0]*100])


# In[29]:


grndf = pd.DataFrame(coexplist,columns=['TF','target','importance'])


# In[30]:


grndf.to_csv('scf_grn_1000.tsv',index=False,sep='\t')


# In[31]:


grndf.head()


# feed into pyscenic

# #bin/bash
# 
# docker run -it --rm \
#     -v /nfs_beijing/:/nfs_beijing/ \
#     aertslab/pyscenic:0.12.1 pyscenic ctx \
#         ./geneemb/scf_grn_1000.tsv \
#         ./geneemb/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather \
#         --annotations_fname ./geneemb/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl \
#         --expression_mtx_fname ./geneemb/data/zheng_subset_cd8t_b_mono.tsv \
#         --mode "custom_multiprocessing" \
#         --output ./geneemb/regulons_1000.csv \
#         --num_workers 40
#         
# docker run -it --rm \
#     -v /nfs_beijing/:/nfs_beijing/ \
#     aertslab/pyscenic:0.12.1 pyscenic aucell \
#         ./geneemb/data/zheng_subset_cd8t_b_mono.tsv \
#         ./geneemb/regulons_1000.csv \
#         -o ./geneemb/auc_mtx_1000.csv \
#         --num_workers 6

# In[ ]:


# from another notebook for post-analysis
# import dependencies
import os
import numpy as np
import pandas as pd
import scanpy as sc
import loompy as lp
import json
import base64
import zlib
from pyscenic.plotting import plot_binarization
from pyscenic.cli.utils import load_signatures
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pyscenic.rss import regulon_specificity_scores
from pyscenic.plotting import plot_rss
from pyscenic.binarization import binarize


# In[3]:


auc_mtx = pd.read_csv('./auc_mtx_1000.csv',index_col=0)


# In[4]:


cellanno = pd.DataFrame(['CD8+ Cytotoxic T']*100+['CD19+ B']*100+['CD14+ Monocyte']*100,columns=['anno'])


# In[6]:


rss_cellType = regulon_specificity_scores( auc_mtx, cellanno['anno'] )
rss_cellType.to_csv('RSS.csv')
rss_cellType


# In[8]:


from adjustText import adjust_text
cats = sorted(list(set(cellanno['anno'])))

fig = plt.figure(figsize=(15, 5))
for c,num in zip(cats, range(1,len(cats)+1)):
    x=rss_cellType.T[c]
    ax = fig.add_subplot(1,3,num)
    plot_rss(rss_cellType, c, top_n=3, max_n=None, ax=ax)
    ax.set_ylim( x.min()-(x.max()-x.min())*0.05 , x.max()+(x.max()-x.min())*0.05 )
    for t in ax.texts:
        t.set_fontsize(12)
    ax.set_ylabel('')
    ax.set_xlabel('')
    adjust_text(ax.texts, autoalign='xy', ha='right', va='bottom', arrowprops=dict(arrowstyle='-',color='lightgrey'), precision=0.001 )
 
fig.text(0.5, 0.0, 'Regulon', ha='center', va='center', size='x-large')
fig.text(0.00, 0.5, 'Regulon specificity score (RSS)', ha='center', va='center', rotation='vertical', size='x-large')
plt.tight_layout()
plt.rcParams.update({
    'figure.autolayout': True,
        'figure.titlesize': 'large' ,
        'axes.labelsize': 'medium',
        'axes.titlesize':'large',
        'xtick.labelsize':'medium',
        'ytick.labelsize':'medium'
        })
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.savefig('RSS.pdf',bbox_inches='tight')


# In[ ]:




