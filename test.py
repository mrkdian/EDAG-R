import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import pickle
import os


# check span2span accuracy
# epoch_range = list(range(1, 23))
# for epoch in epoch_range:
#     print('epoch: %d----------------------------------' % epoch)
#     o = pickle.load(open('Exps_4/HelloEDAG/Output/dee_eval.dev.gold_span.Graph.%d.pkl' % epoch, mode='rb'))

#     for bound in np.arange(0.1, 1, 0.1):
#         total_pred = []
#         total_gt = []
#         for info in o:
#             pred = (info[-1].reshape(-1) > bound).astype(np.int)
#             gt = (info[3].span2span_target > bound).astype(np.int).reshape(-1)
#             total_pred.append(pred)
#             total_gt.append(gt)
#         total_pred = np.hstack(total_pred)
#         total_gt = np.hstack(total_gt)
#         print('bound: %.1f, acc: %.3f' % (bound, accuracy_score(total_gt, total_pred)))



# compare ee f1

ee_types = [
    'EquityFreeze',
    'EquityRepurchase',
    'EquityUnderweight',
    'EquityOverweight',
    'EquityPledge',
    'total'
]

epoch_range = list(range(1, 80))
plot_files = [
    ['Exps_6/HelloEDAG/Output_1/dee_eval.dev.gold_span.Doc2EDAG.%d.json', 'EDAG', 'red'],
    ['Exps_7/HelloEDAG/Output/dee_eval.dev.gold_span.Graph.%d.json', 'EDAG-R', 'blue'],
    ['Exps_6/HelloEDAG/Output/dee_eval.dev.gold_span.GreedyDec.%d.json', 'GreedyDec', 'green'],
    #['Exps/HelloEDAG/Output/dee_eval.dev.pred_span.DCFEE-O.%d.json', 'DCFEE-M', 'yellow'],
    ['Exps_8/HelloEDAG/Output/dee_eval.dev.gold_span.DCFEE-O.%d.json', 'DCFEE', 'black']
]
best_res = []

for e_i, ee_type in enumerate(ee_types):
    text_label = [plot_file[1] for plot_file in plot_files]
    plot_x = [[] for _ in range(len(plot_files))]
    micro_y = [[] for _ in range(len(plot_files))]
    macro_y = [[] for _ in range(len(plot_files))]
    for epoch in epoch_range:
        plot_json = []
        for t in plot_files:
            f = t[0]
            f = f % epoch
            if os.path.exists(f):
                plot_json.append(json.load(open(f, mode='r')))
            else:
                plot_json.append(None)

        for i, p_json in enumerate(plot_json):
            if p_json is None:
                continue
            plot_x[i].append(epoch)
            if e_i < 5:
                micro_y[i].append(p_json[e_i][0]['MicroF1'])
                macro_y[i].append(p_json[e_i][0]['MacroF1'])
            else:
                micro_y[i].append(p_json[e_i]['MicroF1'])
                macro_y[i].append(p_json[e_i]['MacroF1'])

    plt.cla()
    handles = []
    for i, (_, label, color) in enumerate(plot_files):
        handles.append(plt.plot(plot_x[i], micro_y[i], color=color, label=label)[0])
        if e_i >= 5:
            best_res.append((max(micro_y[i]), label))
    plt.xlabel('epoch')
    plt.ylabel('mirco-f1')
    #plt.xticks(epoch_range, size=1)
    plt.legend(handles, text_label)
    plt.savefig('./%s-micro.png' % ee_type)
    plt.cla()

    handles = []
    for i, (_, label, color) in enumerate(plot_files):
        handles.append(plt.plot(plot_x[i], macro_y[i], color=color, label=label)[0])
    plt.xlabel('epoch')
    plt.ylabel('marco-f1')
    #plt.xticks(epoch_range, size=1)
    plt.legend(handles, text_label)
    plt.savefig('./%s-macro.png' % ee_type)
    plt.cla()
print(best_res)


# # ner
plot_files = [
    #['Exps_6/HelloEDAG/Output_1/dee_ner.dev.pred_span.Doc2EDAG.%d.json', 'Doc2EDAG', 'red'],
    #['Exps_7/HelloEDAG/Output/dee_ner.dev.pred_span.Graph.%d.json', 'Graph', 'blue'],
    ['Exps_8/HelloEDAG/Output/dee_ner.dev.pred_span.DCFEE-O.%d.json', 'ner', 'blue'],
    ['Exps_9/HelloEDAG/Output/dee_ner.dev.pred_span.DCFEE-O.%d.json', 'ner-R', 'red'],
    #['Exps_9/HelloEDAG/Output/dee_ner.dev.pred_span.DCFEE-O.%d.json', 'ner', 'blue']
]
text_label = [plot_file[1] for plot_file in plot_files]
plot_x = [[] for _ in range(len(plot_files))]
micro_y = [[] for _ in range(len(plot_files))]
macro_y = [[] for _ in range(len(plot_files))]
best_res = [[-1, 0, -1, 0] for _ in plot_files]
for epoch in epoch_range:
    plot_json = []
    for t in plot_files:
        f = t[0]
        f = f % epoch
        if os.path.exists(f):
            plot_json.append(json.load(open(f, mode='r')))
        else:
            plot_json.append(None)

    for i, p_json in enumerate(plot_json):
        if p_json is None:
            continue
        plot_x[i].append(epoch)
        micro_y[i].append(p_json['micro_f1'])
        macro_y[i].append(p_json['macro_f1'])

        if p_json['micro_f1'] > best_res[i][0]:
            best_res[i][0] = p_json['micro_f1']
            best_res[i][1] = epoch
        if p_json['macro_f1'] > best_res[i][2]:
            best_res[i][2] = p_json['macro_f1']
            best_res[i][3] = epoch
plt.cla()
handles = []
for i, (_, label, color) in enumerate(plot_files):
    handles.append(plt.plot(plot_x[i], micro_y[i], color=color, label=label)[0])
plt.xlabel('epoch')
plt.ylabel('mirco-f1')
#plt.xticks(epoch_range, size=1)
plt.legend(handles, text_label)
plt.savefig('./micro-ner.png')
plt.cla()

handles = []
for i, (_, label, color) in enumerate(plot_files):
    handles.append(plt.plot(plot_x[i], macro_y[i], color=color, label=label)[0])
plt.xlabel('epoch')
plt.ylabel('marco-f1')
#plt.xticks(epoch_range, size=1)
plt.legend(handles, text_label)
plt.savefig('./macro-ner.png')
plt.cla()

for i, (_, label, _) in enumerate(plot_files):
    print(label, ':', best_res[i])
