from fastai.imports import *
from matplotlib import patches, patheffects
from fastai.dataset import *

def wh_bb(a): return np.array([a[1], a[0], a[3]+a[1]-1, a[2]+a[0]-1])
def bb_wh(a): return np.array([a[1],a[0],a[3]-a[1]+1,a[2]-a[0]+1])

def show_img(im, figsize=None, ax=None, grid=False):
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    if grid:
        ax.set_xticks(np.linspace(0, 224, 9))
        ax.set_yticks(np.linspace(0, 224, 9))
        ax.grid()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    else:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)        
    return ax

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(linewidth=lw, foreground='black'), 
                        patheffects.Normal()])

def draw_rect(ax, b, color='white'):
    patch = ax.add_patch(patches.Rectangle(
        b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
    draw_outline(patch, 4)

def draw_text(ax, xy, txt, sz=14, color='white'):
    text = ax.text(*xy, txt, verticalalignment='top', 
                   color=color, fontsize=sz, weight='bold')
    draw_outline(text, 1)

class ConcatLblDataset(Dataset):
    def __init__(self, ds, y2): 
        self.ds, self.y2 = ds, y2
        self.sz = ds.sz
    
    def __len__(self): return len(self.ds)
    
    def __getitem__(self, i):
        x, y = self.ds[i]
        return (x, (y, self.y2[i]))

def append_y(md, md2):
    md.trn_dl.dataset = ConcatLblDataset(md.trn_ds, md2.trn_y)
    md.val_dl.dataset = ConcatLblDataset(md.val_ds, md2.val_y)
    
class VOCData:    
    def __init__(self, fname, img_path):
        j = json.load(Path(fname).open())
        self.img_path = img_path
        self.cats = {o['id']: o['name'] for o in j['categories']}
        self.ids = [o['id'] for o in j['images']]
        self.fnames = {o['id']: o['file_name'] for o in j['images']}
        self.anns = collections.defaultdict(lambda:[])
        for o in j['annotations']: 
            if not o['ignore']:
                b, c = wh_bb(o['bbox']), o['category_id']
                self.anns[o['image_id']].append((b, c))
        self._lrg_anns = None
        
    @property
    def lrg_anns(self):
        if self._lrg_anns is None:
            largest = lambda ann: max(ann, key=lambda x: np.product(x[0][-2:]-x[0][:2]))
            self._lrg_anns = {id: largest(ann) for id,ann in self.anns.items()}
        return self._lrg_anns

    def open_img(self, id):
        return open_image(self.img_path/self.fnames[id])
    
    def show(self, id, figsize=(16,8), anns=None, largest=False):
        if anns is None: 
            anns = [self.lrg_anns[id]] if largest else self.anns[id]
        im = self.open_img(id)
        print(im.shape)
        ax = show_img(im, figsize=figsize)
        for b,c in anns:
            b = bb_wh(b)
            draw_rect(ax, b)
            draw_text(ax, b[:2], self.cats[c], sz=16)
            
    def show_batch(self, x, y):
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        for i, ax in enumerate(axes.flat):
            bb = [bb_wh(o) for o in y[0][i].reshape(-1,4)]
            ax = show_img(x[i], ax=ax)
            for b, c in zip(bb, y[1][i]):
                if (b[2] > 1):
                    draw_rect(ax, b)
                    draw_text(ax, b[:2], self.cats[int(c)])
        plt.tight_layout()
    
    def get_cls_df(self, largest=False, dedupe=False, named=False):
        fns = [self.fnames[i] for i in self.ids]
        if largest:
            cts = [self.cats[self.lrg_anns[i][1]] for i in self.ids]
        else:
            wrap = set if dedupe else list
            ct_ids = [wrap([a[1] for a in self.anns[i]]) for i in self.ids]
            cts = [' '.join(self.cats[ci] if named else str(ci) for ci in ids) for ids in ct_ids]
        return pd.DataFrame({'fn': fns, 'cat': cts}, columns=['fn', 'cat'])
        
    def get_bb_df(self, largest=False):
        fns = [self.fnames[i] for i in self.ids]
        if largest:
            bbs = np.array([' '.join(str(p) for p in self.lrg_anns[i][0]) for i in self.ids])
        else:
            _bbs = [np.concatenate([a[0] for a in self.anns[i]]) for i in self.ids]
            bbs = [' '.join(str(p) for p in a) for a in _bbs]
        return pd.DataFrame({ 'fn': fns, 'bbox': bbs}, columns=['fn', 'bbox'])
