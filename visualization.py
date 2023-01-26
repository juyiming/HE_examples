import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import join as oj
import math
import matplotlib.colors as colors
import pickle



def visualize_scores(scores, label, text_orig, score_orig, sweep_dim=1, method='break_down'):
    plt.figure(figsize=(2, 10))
    try:
        p = scores.data.cpu().numpy()[:, label]
    except:
        p = scores

    # plot with labels
    text_orig = text_orig[sweep_dim - 1:]  # todo - don't do this, deal with edges better
    plt.barh(range(p.size), p[::-1], align='center', tick_label=text_orig[::-1])
    c = "pos" if label == 0 else "neg"
    plt.title(method + ' class ' + c + '\n(higher is more important)')  # pretty sure 1 is positive, 2 is negative




def print_scores(lists, text_orig, num_iters):
    text_orig = np.array(text_orig)
    print('score_orig', lists['score_orig'])

    print(text_orig)
    print(lists['scores_list'][0])

    # print out blobs and corresponding scores
    for i in range(1, num_iters):
        print('iter', i)
        comps = lists['comps_list'][i]
        comp_scores_list = lists['comp_scores_list'][i]

        # sort scores in decreasing order
        comps_with_scores = sorted(zip(range(1, np.max(comps) + 1),
                                       [comp_scores_list[i] for i in comp_scores_list.keys()]),
                                   key=lambda x: x[1], reverse=True)

        for comp_num, comp_score in comps_with_scores:
            print(comp_num, '\t%.3f, %s' % (comp_score, str(text_orig[comps == comp_num])))


def word_heatmap(text_orig, lists, plot_id, label_pred, label, method=None, subtract=True, data=None, fontsize=9):
    text_orig = np.array(text_orig)
    num_words = text_orig.size

    num_iters = len(lists)

    data = lists
    text_len = len(' '.join(text_orig))

    size = int(text_len/5)
    if size < 16:
        size = 16
    if num_iters == 1:
        plt.figure(figsize=(size, 1), dpi=300)
    else:
        plt.figure(figsize=(size, int(size/5)), dpi=300)

    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))


    cmap = plt.get_cmap('RdBu')

    abs_lim = max(abs(np.nanmax(data)), abs(np.nanmin(data)))
    c = plt.pcolor(data,
                   edgecolors='k',
                   linewidths=0,
                   norm=MidpointNormalize(vmin=abs_lim * -1, midpoint=0., vmax=abs_lim),
                   cmap=cmap)

    def show_values(pc, text_orig, data, fontsize, fmt="%s", **kw):
        val_mean = np.nanmean(data)
        val_min = np.min(data)
        pc.update_scalarmappable()
        # ax = pc.get_axes()
        ax = pc.axes

        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            # pick color for text
            if np.all(color[:3] > 0.5):  # value > val_mean: #value > val_mean: #
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            x_ind = math.floor(x)
            y_ind = math.floor(y)

            # sometimes don't display text
            if y_ind == 0 or data[y_ind, x_ind] != 0:  # > val_min:
                ax.text(x, y, fmt % text_orig[x_ind],
                        ha="center", va="center",
                        color=color, fontsize=fontsize, **kw)

    show_values(c, text_orig, data, fontsize)
    cb = plt.colorbar(c, extend='both')  # fig.colorbar(pcm, ax=ax[0], extend='both')
    cb.outline.set_visible(False)
    plt.xlim((0, num_words))
    plt.ylim((0, num_iters))
    plt.yticks([])
    plt.plot([0, num_words], [1, 1], color='black')
    plt.xticks([])
    plt.title('prediction:  ' + label_pred + '     text:  ' + ' '.join(text_orig))

    cb.ax.set_title('attribution score')
    plt.savefig('{}/{}'.format(ouput_dir,plot_id))#,bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    data_dir = ''    # he data path
    label_dir = ''   # prediction path
    ouput_dir = ''


    with open(data_dir,'rb') as f:
        target = pickle.load(f)
    with open(label_dir, 'rb') as f:
        labels = pickle.load(f)


    plot_id = 0
    for prelabel, item in zip(labels, target):
        if len(item[2]) == 1:
            indexes_list, score_list, words_ori = item
            l = len(words_ori)
            matrix = np.zeros((l,l))
            mask = np.ones((l,l))
            ids_old = [None for x in range(l)]
            for i, x in enumerate(zip(indexes_list, score_list)):
                ids, socres = x

                for j,y in enumerate(zip(ids, socres)):
                    id, score = y
                    if i != 0:
                        if id != ids_old[j]:
                            for idd in id:
                                matrix[i][idd] = score #*len(id)
                                mask[i][idd] = 0
                            ids_old = ids
                            break
                    else:
                        for idd in id:
                            matrix[i][idd] = score #*len(id)
                            mask[i][idd] = 0
                        ids_old = ids
        if len(item[2]) == 2:
            indexes_list, score_list, words_ori = item
            words_ori_a, words_ori_b = words_ori
            l_a = len(words_ori_a)

            l = len(words_ori_a) + len(words_ori_b) + 1
            words_ori = []
            words_ori.extend(words_ori_a)
            words_ori.append('[SEP]')
            words_ori.extend(words_ori_b)

            matrix = np.zeros((l, l))
            mask = np.ones((l, l))
            ids_old = [None for x in range(l)]
            for i, x in enumerate(zip(indexes_list, score_list)):
                ids, socres = x

                for j, y in enumerate(zip(ids, socres)):
                    id, score = y
                    if i != 0:
                        if id != ids_old[j]:
                            for idd in id:

                                matrix[i][idd] = score
                                mask[i][idd] = 0
                            ids_old = ids
                            break
                    else:
                        for idd in id:
                            matrix[i][idd] = score
                            mask[i][idd] = 0
                        ids_old = ids
        if len(item[2]) == 1:
            if prelabel == 1:
                label = 'positive'
            else:
                label = 'negative'
        else:
            label_list = ["contradiction", "entailment", "neutral"]
            label = label_list[prelabel]

        word_heatmap(words_ori, matrix, plot_id, label, 1, fontsize=9)
        plot_id += 1
        print('finished')
