import os
import numpy as np
# from utils.class_id_map import get_class2id_map
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm 
from matplotlib import axes
# import seaborn as sns
# from sklearn import manifold
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

def color_map_color(value, cmap_name='hsv', vmin=0, vmax=20):
    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    color = matplotlib.colors.rgb2hex(rgb)
    return color

## TSNE 特征降维 ##
def Tsne(feature, sample, CONFIG):
    output_photo_path = CONFIG.result_path+'/split'+str(CONFIG.split)+'/'+str(CONFIG.dataset)+'_split'+str(CONFIG.split)+'_'+'tsne_photo/'

    if not os.path.exists(output_photo_path):
        os.makedirs(output_photo_path)

    feature = feature.squeeze(0).cpu().numpy().transpose(1,0)
    t = sample["label"]

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # tsne = manifold.TSNE()
    feature = tsne.fit_transform(feature)
    x_min, x_max = np.min(feature, 0), np.max(feature, 0)
    feature = (feature - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(50,50), dpi=200)#设置画布的尺寸

    print(feature.shape)
    # plt.scatter(feature[:,0], feature[:,1], c= t)
    for i in range(0,feature.shape[0],5):

        plt.text(feature[i,0], feature[i,1], s=str(i), fontsize=20, color=color_map_color(t[0,i]))

    plt.savefig(output_photo_path + str(sample['feature_path']).split(".")[0].split("/")[-1]+'.png')
    plt.close(fig)

def output_label_id(pred, sample, CONFIG):

    CONFIG.output_label_path = CONFIG.result_path+'/split'+str(CONFIG.split)+'/'+str(CONFIG.dataset)+'_split'+str(CONFIG.split)+'_'+'prediction/'
    print(CONFIG.output_label_path)
    if not os.path.exists(CONFIG.output_label_path):
        os.makedirs(CONFIG.output_label_path)
    
    

    f_ptr = open(CONFIG.output_label_path+ str(sample['feature_path']).split(".npy")[-2].split("/")[-1]+'.txt', "w")
    for label in pred:
        f_ptr.write(''.join(str(label)))
        f_ptr.write('\n')
    f_ptr.close()
    

def output_label_class(pred, sample, CONFIG):

    CONFIG.output_label_path = CONFIG.result_path +'/output_label_class/'
    if not os.path.exists(CONFIG.output_label_path):
        os.makedirs(CONFIG.output_label_path)

    with open(os.path.join(CONFIG.dataset_dir, "{}/mapping.txt".format(CONFIG.dataset)), 'r') as f:
        actions = f.read().split('\n')[:-1]

    id2class_map = dict()
    for a in actions:
        id2class_map[a.split()[0]] = a.split()[1]
    print(id2class_map)
    f_ptr = open(CONFIG.output_label_path+ str(sample['feature_path']).split(".")[0].split("/")[-1]+'.txt', "w")
    for label in pred:
        label = id2class_map[str(label)]
        f_ptr.write(''.join(str(label)))
        f_ptr.write('\n')
    f_ptr.close()


def draw_boundary_2(pred, gt):

    photo_path = '/home/liyunheng/Segmentation/asrf-main/result/50salads/asrf/yuanshifenli/split1/photo_boundary/'
    if not os.path.exists(photo_path):
        os.makedirs(photo_path)


    plt.figure(figsize=(10,10), dpi=300)#设置画布的尺寸

    plt.plot(np.arange(len(pred)), list(pred), 'r:')
    # plt.plot(np.arange(len(gt)), list(gt), color = 'blue')
    plt.plot(np.arange(len(gt)), list(gt), 'g--' )


    plt.savefig(photo_path + str(pred.shape) +'.png')

def draw_boundary_3(pred, gt, refind_pred):

    photo_path = './photo_boundary/'
    if not os.path.exists(photo_path):
        os.makedirs(photo_path)


    plt.figure(figsize=(10,10), dpi=300)#设置画布的尺寸

    plt.plot(np.arange(len(pred)), list(pred), 'r:')
    plt.plot(np.arange(len(gt)), list(gt), color = 'blue')
    plt.plot(np.arange(len(refind_pred)), list(refind_pred), 'g--' )


    plt.savefig(photo_path + str(pred.shape) +'.png')


def draw_label_2(output_cls, sample, refind_pred, refind_pred_LBS, CONFIG):

    CONFIG.photo_path = CONFIG.result_path +'/split'+ str(CONFIG.split) +'/photo_label/'
    if not os.path.exists(CONFIG.photo_path):
        os.makedirs(CONFIG.photo_path)

    gt = sample['label'].data.cpu().squeeze(0).numpy()
    label_pred = np.argmax(output_cls, axis=1).squeeze()
    pro_pred2 = np.max(output_cls, axis=1).squeeze()
    output_cls = np.exp(output_cls) / np.sum(np.exp(output_cls), axis=1)
    pro_pred = np.max(output_cls, axis=1).squeeze()

    wrong_label = np.abs(gt - refind_pred_LBS)
    wrong_label[wrong_label>0] = -1

    wrong_refinf_label = np.abs(gt - refind_pred)
    wrong_refinf_label[wrong_refinf_label>0] = -1

    plt.figure(figsize=(20,20), dpi=300)#设置画布的尺寸

    plt.plot(np.arange(len(label_pred)), list(label_pred), 'r:')
    plt.plot(np.arange(len(gt)), list(gt), color = 'blue')
    plt.plot(np.arange(len(refind_pred)), list(refind_pred), 'g--' )
    plt.plot(np.arange(len(refind_pred_LBS)), list(refind_pred_LBS) )
    plt.plot(np.arange(len(wrong_label)), list(wrong_label-1) )
    plt.plot(np.arange(len(wrong_refinf_label)), list(wrong_refinf_label-3) )

    # plt.plot(np.arange(len(pro_pred)), list(pro_pred-2) )
    # plt.plot(np.arange(len(pro_pred2)), list(pro_pred2-10) )
    # plt.plot(np.arange(len(wrong_label)), list(wrong_label-1) )
    # plt.plot(np.arange(len(wrong_refinf_label)), list(wrong_refinf_label-5) )

    plt.savefig(CONFIG.photo_path + str(sample['feature_path']).split(".")[0].split("/")[-1]+ str(label_pred.shape)+'.png')


def draw_label_3(pred, label_submax, refind_pred, index, index2, sample, boundary_confidience, CONFIG):

    CONFIG.photo_path = CONFIG.result_path +'/photo_label/'
    if not os.path.exists(CONFIG.photo_path):
        os.makedirs(CONFIG.photo_path)

    gt = sample['label'].data.cpu().squeeze(0).numpy()
    plt.figure(figsize=(10,20), dpi=300)#设置画布的尺寸
    
    for i, ind in enumerate(boundary_confidience):
    #     plt.axvline(i)

        plt.text(index[i]-50,i*0.5, np.around(ind,1),size = 5)
    
  

    # new_texts =[plt.text(index[i],i, np.around(ind,1),size = 10) for i, ind in enumerate(boundary_confidience)]

    # #     new_texts.append(plt.text(index[i],-1, np.around(ind,1),size = 10))
    # adjust_text(new_texts, 
    #             arrowprops=dict(arrowstyle='->', 
    #                             color='red',
    #                             lw=1))


    for i,ind in enumerate(index):
    #     plt.axvline(i)
        plt.vlines(ind, -0.5, 17, colors='y', alpha=0.2)

    for i in index2:
    #     plt.axvline(i)
        plt.vlines(i, -1, -0.5, colors='k')
    plt.subplot(211) 
    plt.plot(np.arange(len(pred)), list(pred), 'r:')

    plt.plot(np.arange(len(gt)), list(gt), color = 'blue')
    plt.plot(np.arange(len(refind_pred)), list(refind_pred), 'g--' )

    plt.subplot(212)
    plt.plot(np.arange(len(gt)), list(gt), color = 'blue')
    plt.plot(np.arange(len(label_submax)), list(label_submax), 'r:')
    

    plt.savefig(CONFIG.photo_path + str(sample['feature_path']).split(".")[0].split("/")[-1]+ str(pred.shape)+'.png')


def draw_label_4(output_cls, sample, refind_pred, CONFIG):

    CONFIG.photo_path = CONFIG.result_path +'/split'+ str(CONFIG.split) +'/photo_label/'
    if not os.path.exists(CONFIG.photo_path):
        os.makedirs(CONFIG.photo_path)

    gt = sample['label'].data.cpu().squeeze(0).numpy()
    label_pred = np.argmax(output_cls, axis=1).squeeze()
    # pro_pred2 = np.max(output_cls, axis=1).squeeze()
    # output_cls = np.exp(output_cls) / np.sum(np.exp(output_cls), axis=1)
    pro_pred = np.max(output_cls, axis=1).squeeze()

    wrong_refinf_label = np.abs(gt - refind_pred)
    wrong_refinf_label[wrong_refinf_label>0] = -1

    plt.figure(figsize=(20,20), dpi=300)#设置画布的尺寸

    plt.plot(np.arange(len(label_pred)), list(label_pred), 'r:')
    plt.plot(np.arange(len(gt)), list(gt), color = 'blue')
    plt.plot(np.arange(len(refind_pred)), list(refind_pred), 'g--' )
    plt.plot(np.arange(len(wrong_refinf_label)), list(wrong_refinf_label-1) )
    
    plt.axhline(-15+6)
    plt.plot(np.arange(len(pro_pred)), list(pro_pred-15) )
    # plt.plot(np.arange(len(pro_pred2)), list(pro_pred2-10) )
    # plt.plot(np.arange(len(wrong_label)), list(wrong_label-1) )
    # plt.plot(np.arange(len(wrong_refinf_label)), list(wrong_refinf_label-5) )

    plt.savefig(CONFIG.photo_path + str(sample['feature_path']).split(".")[0].split("/")[-1]+ str(label_pred.shape)+'.png')


# def draw_label_3(pred, refind_pred, sample, CONFIG):

#     CONFIG.photo_path = CONFIG.result_path +'/photo_label/'
#     if not os.path.exists(CONFIG.photo_path):
#         os.makedirs(CONFIG.photo_path)

#     gt = sample['label'].data.cpu().squeeze(0).numpy()
#     plt.figure(figsize=(10,10), dpi=300)#设置画布的尺寸

#     plt.plot(np.arange(len(pred)), list(pred), 'r:')
#     plt.plot(np.arange(len(gt)), list(gt), color = 'blue')
#     plt.plot(np.arange(len(refind_pred)), list(refind_pred), 'g--' )

#     plt.savefig(CONFIG.photo_path + str(sample['feature_path']).split(".")[0].split("/")[-1]+'.png')

def draw_confusion_matrix(data, acc, edit_score, f1s, CONFIG):

    data = np.around(data/sum(data), 2)
    classes=get_class2id_map(CONFIG.dataset, dataset_dir=CONFIG.dataset_dir)

    figure=plt.figure(facecolor='w',figsize=(30, 30))
    ax=figure.add_subplot(1,1,1)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes,fontsize=30)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes,rotation=-60, verticalalignment='baseline',fontsize=30)
    ax.xaxis.set_ticks_position('top')

    ax.set_title(str(np.around([acc, edit_score, f1s[0], f1s[1], f1s[2]],2)), verticalalignment='bottom', fontsize=50)

    cmap=cm.Blues
    # cmap=cm.get_cmap('rainbow',1000)
    map=ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto',vmin=0,vmax=1)
    # cb=ax.colorbar(mappable=map,cax=None,ax=None,shrink=0.5) #图例

    for i in range(len(data)):
        for j in range(len(data[i])):
            if(data[i,j]<0.4):
                ax.annotate(data[i,j], xy=(i, j), horizontalalignment='center', verticalalignment='center',fontsize=20)
            else:
                ax.annotate(data[i,j], xy=(i, j), horizontalalignment='center', color='white',verticalalignment='center',fontsize=20)

    ax.savefig(CONFIG.dataset + '_' + str(CONFIG.split) +'.jpg')



# def draw_feature(data):

#     data = np.around(data/sum(data), 2)
#     figure=plt.figure(facecolor='w',figsize=(30, 30))
#     ax=figure.add_subplot(1,1,1)
  
#     cmap=cm.Blues
#     # cmap=cm.get_cmap('rainbow',1000)
#     map=ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto',vmin=0,vmax=1)
#     cb=ax.colorbar(mappable=map,cax=None,ax=None,shrink=0.5) #图例

#     ax.savefig(str(data.shape) +'.jpg')    
    

def draw_Statistics(g_counter, p_counter, acc, edit_score, f1s): 

    s = sorted(g_counter.items(), key=lambda x: x[1], reverse=False)
    s2 = sorted(p_counter.items(), key=lambda x: x[1], reverse=False)

    x_x = []
    y_y = []
    for i in s:
        x_x.append(i[0]-2)
        y_y.append(i[1])

    x = x_x
    y = y_y

    x_x2 = []
    y_y2 = []
    for i in s2:
        x_x2.append(i[0]+2)
        y_y2.append(i[1])
    x2 = x_x2
    y2 = y_y2

    figure=plt.figure(facecolor='w',figsize=(50, 50))
    ax=figure.add_subplot(1,1,1)
    width=3
    fig, ax = plt.subplots()

    y_label = []
    for i in range (0,160,10):
        y_label.append(i)
    
    ax.set_yticks(y_label)
    ax.barh(x, y, height=width, label='True', color="deepskyblue")
    ax.barh(x2, y2, height=width, label='Predict',color='red')

    ax.legend(["Predict"],loc="lower right")
    ax.legend(["True"],loc="lower right")
    ax.legend(loc="lower right")  # 防止label和图像重合显示不出来

    for a, b in zip(x, y):
        ax.text((b+int(len(x)/5)), a, b, ha='center', va='center',fontsize=7)
    for a, b in zip(x2, y2):
        ax.text((b+len(x2)/5), a, b, ha='center', va='center',fontsize=7)

    ax.set_title(str(np.around([acc, edit_score, f1s[0], f1s[1], f1s[2]],2)))

    plt.savefig('./draw_Statistics.jpg')



def draw_openpose(data2,x,output_path = './AAFDGAT_drop0/'):
    data3 = np.array(data2.clone().data.cpu()).squeeze()
    data = np.array(data2.clone().data.cpu())[0].squeeze()
    # for i in range(0,3):
    #     data += data3[i]
    # data = np.array(data2.clone().data.cpu()).squeeze()

    data = np.around(data, 4)


    figure=plt.figure(facecolor='w',figsize=(30, 30))
    ax=figure.add_subplot(1,1,1)
    plt.tick_params(labelsize=30)
    ax.set_yticks(range(25))
  
    ax.set_xticks(range(25))

    ax.xaxis.set_ticks_position('top')


    cmap=cm.coolwarm
    # cmap=cm.get_cmap('rainbow',1000)

    map=sns.heatmap(data, cmap=cmap, center=0, cbar =False, annot=True,annot_kws={'size':15,'weight':'bold'})
    # cb=ax.colorbar(mappable=map,cax=None,ax=None,shrink=0.5) #图例
    # map=ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto')


    # for i in range(len(data)):
    #     for j in range(len(data[i])):
    #         if(data[i,j]<0):
    #             ax.annotate(data[i,j], xy=(i, j), horizontalalignment='center', verticalalignment='center',fontsize=20)
    #         else:
    #             ax.annotate(data[i,j], xy=(i, j), horizontalalignment='center',verticalalignment='center',fontsize=20)
    fig = plt.gcf()
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fig.savefig(output_path+str(x.shape[2])+'.jpg')

def draw_graph(data2,x):
    _edges =[(2,9), (1,2), (16,1), (18,16), (17,1), (19,17), (6,2),
                    (7,6), (8,7), (3,2), (4,3), (5,4), (10,9),
                    (11, 10), (12, 11), (25, 12), (23, 12), (24, 23), (13,9),
                    (14, 13), (15, 14), (22, 15), (20, 15), (21, 20),
                    ]
    edges = []
    for edge in _edges:
        edges.append((edge[0]-1,edge[1]-1))

    data = np.array(data2.clone().data.cpu())[0][0].squeeze()
    # data = np.array(data2.clone().data.cpu()).squeeze()
    x=np.array(x.clone().data.cpu()).squeeze()

    for i in range(200,x.shape[1]):
        array = np.zeros((2,25))
        array[0,:] = -x[0,i:i+1,:]
        array[1,:] = -x[1,i:i+1,:]

        # data = np.around(data, 2)

        figure=plt.figure(facecolor='w',figsize=(30, 30))

        ax = plt.subplot()
        plt.tick_params(labelsize=50)

        ax.plot(array[0,:], array[1,:], 'o',lw =10)  
        for edge in edges:
            if edge[0] and edge[1]:
                x1 = array[:,edge[0]]
                y1 = array[:,edge[1]]
                p1 = [x1[0],y1[0]]
                p2 = [x1[1],y1[1]]
                ax.plot(p1,p2,'b-',lw =10)
        fig = plt.gcf()
        output_path = './SFGAT5_attention/'
        # output_path = './SFGAT5/'

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        fig.savefig(output_path+str(i)+'.jpg')


def draw_flowformer_map(x, photo_path ='./photo_feature/'):
    # print(x.shape)

    if not os.path.exists(photo_path):
        os.makedirs(photo_path)

    # x = x.unsqueeze(0)

    # feature = x[:, :, :2000]
    # feature = x[:, 4:, :]
    feature = x[:, :, :]

    N, C, T = feature.size()

    data = np.array(feature.clone().data.cpu()).squeeze(0)

    # data = np.around(data, 4)

    print("figure: draw_feature")
    figure=plt.figure(facecolor='w',figsize=(300, 100))
    ax=figure.add_subplot(1,1,1)
    plt.tick_params(labelsize=50)
    ax.set_yticks(range(C))
    # ax.set_xticks(range(T))
    cmap=cm.coolwarm
    
    # cmap=cm.get_cmap('rainbow',1000)

    # map=sns.heatmap(data, cmap=cmap)
    # cb=ax.colorbar(mappable=map,cax=None,ax=None,shrink=0.5) #图例

    im=ax.imshow(data,cmap=cmap,aspect='auto')#cmap='Greys'),设置热力图颜色，一般默认为蓝黄色，aspect='auto'热力块大小随着画布大小自动变化，如果不设置的话为方框。
    #create colorbar色条
    cbar=ax.figure.colorbar(im, ax=ax)
    #colorbar的设置
    cbar.ax.set_ylabel('score', rotation=-90, va="bottom",fontsize=120,fontname='Times New Roman')#colorbar标签为‘score’，纵向放置，字体大小为18，字体为新罗马字体
    #color色条本身上刻度值大小及字体设置
    # cbar.ax.yaxis.set_major_locator(MultipleLocator(1))
    # cbar.ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    cbar.ax.tick_params(labelsize=120)
    cbarlabels = cbar.ax.get_yticklabels() 
    [label.set_fontname('Times New Roman') for label in cbarlabels]

    plt.savefig(photo_path + str(x.shape)+'.png', bbox_inches = 'tight')
    plt.close(figure)

    print("done")


def draw_flowformer_encoder_feature(x, photo_path ='./photo_feature/'):
    # print(x.shape)

    if not os.path.exists(photo_path):
        os.makedirs(photo_path)

    # x = x.unsqueeze(0)

    # feature = x[:, :, :2000]
    feature = x[:, 4:, :]
    feature = feature[:, ::8, :]
    N, C, T = feature.size()

    data = np.array(feature.clone().data.cpu()).squeeze(0)

    # data = np.around(data, 4)

    print("figure: draw_feature")
    figure=plt.figure(facecolor='w',figsize=(300, 100))
    ax=figure.add_subplot(1,1,1)
    plt.tick_params(labelsize=50)
    ax.set_yticks(range(C))
    # ax.set_xticks(range(T))
    cmap=cm.coolwarm
    
    # cmap=cm.get_cmap('rainbow',1000)

    # map=sns.heatmap(data, cmap=cmap)
    # cb=ax.colorbar(mappable=map,cax=None,ax=None,shrink=0.5) #图例

    im=ax.imshow(data,cmap=cmap,aspect='auto')#cmap='Greys'),设置热力图颜色，一般默认为蓝黄色，aspect='auto'热力块大小随着画布大小自动变化，如果不设置的话为方框。
    #create colorbar色条
    cbar=ax.figure.colorbar(im, ax=ax)
    #colorbar的设置
    cbar.ax.set_ylabel('score', rotation=-90, va="bottom",fontsize=120,fontname='Times New Roman')#colorbar标签为‘score’，纵向放置，字体大小为18，字体为新罗马字体
    #color色条本身上刻度值大小及字体设置
    # cbar.ax.yaxis.set_major_locator(MultipleLocator(1))
    # cbar.ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    cbar.ax.tick_params(labelsize=120)
    cbarlabels = cbar.ax.get_yticklabels() 
    [label.set_fontname('Times New Roman') for label in cbarlabels]

    plt.savefig(photo_path + str(x.shape)+'.png', bbox_inches = 'tight')
    plt.close(figure)

    print("done")

def eval_draw_flowformer_map(map, sample, CONFIG):
    output_photo_path = CONFIG.result_path+'/split'+str(CONFIG.split)+'/'+str(CONFIG.dataset)+'_split'+str(CONFIG.split)+'_'+'encoder_map_photo/'
    if not os.path.exists(output_photo_path):
        os.makedirs(output_photo_path)

    boundary = sample["boundary"].squeeze().to("cpu").data.numpy()
    map = map[:,64:,:]
    N, C, T = map.size()
    data = np.array(map.clone().data.cpu()).squeeze(0)

    print("figure: draw_feature", str(sample['feature_path']).split(".")[0].split("/")[-1])
    figure=plt.figure(facecolor='w',figsize=(300, 100))
    ax=figure.add_subplot(1,1,1)
    plt.tick_params(labelsize=50)
    ax.set_yticks(range(C))
    ax.set_xticks(range(0, T, 100))
    cmap=cm.coolwarm
    # cmap=cm.get_cmap('rainbow',1000)
    # map=sns.heatmap(data, cmap=cmap)
    # cb=ax.colorbar(mappable=map,cax=None,ax=None,shrink=0.5) #图例
    im=ax.imshow(data,cmap=cmap,aspect='auto')#cmap='Greys'),设置热力图颜色，一般默认为蓝黄色，aspect='auto'热力块大小随着画布大小自动变化，如果不设置的话为方框。
    #create colorbar色条
    cbar=ax.figure.colorbar(im, ax=ax)
    #colorbar的设置
    cbar.ax.set_ylabel('score', rotation=-90, va="bottom",fontsize=120,fontname='Times New Roman')#colorbar标签为‘score’，纵向放置，字体大小为18，字体为新罗马字体
    #color色条本身上刻度值大小及字体设置
    cbar.ax.tick_params(labelsize=120)
    cbarlabels = cbar.ax.get_yticklabels() 
    [label.set_fontname('Times New Roman') for label in cbarlabels]
    
    bound = []
    for i, ind in enumerate(boundary):
        if (ind!=0):
            plt.axvline(i,color='black',linewidth = '3')
            bound.append(i)
    print("boundary", bound)

    plt.savefig(output_photo_path + str(sample['feature_path']).split(".")[0].split("/")[-1]+'.png', bbox_inches = 'tight')

    plt.close(figure)
    
    print("done")

def eval_draw_flowformer_encoder_feature(dict_encoder, sample, CONFIG):
    # print(x.shape)
    output_photo_path = CONFIG.result_path+'/split'+str(CONFIG.split)+'/'+str(CONFIG.dataset)+'_split'+str(CONFIG.split)+'_'+'encoder_feature_photo/'
    if not os.path.exists(output_photo_path):
        os.makedirs(output_photo_path)

    boundary = sample["boundary"].squeeze().to("cpu").data.numpy()
    feature = dict_encoder[:, ::8, :]
    N, C, T = feature.size()
    data = np.array(feature.clone().data.cpu()).squeeze(0)

    print("figure: draw_feature", str(sample['feature_path']).split(".")[0].split("/")[-1])
    figure=plt.figure(facecolor='w',figsize=(300, 100))
    ax=figure.add_subplot(1,1,1)
    plt.tick_params(labelsize=50)
    ax.set_yticks(range(C))
    ax.set_xticks(range(0, T, 100))
    cmap=cm.coolwarm
    # cmap=cm.get_cmap('rainbow',1000)
    # map=sns.heatmap(data, cmap=cmap)
    # cb=ax.colorbar(mappable=map,cax=None,ax=None,shrink=0.5) #图例
    im=ax.imshow(data,cmap=cmap,vmax=4,vmin=-4,aspect='auto')#cmap='Greys'),设置热力图颜色，一般默认为蓝黄色，aspect='auto'热力块大小随着画布大小自动变化，如果不设置的话为方框。
    #create colorbar色条
    cbar=ax.figure.colorbar(im, ax=ax)
    #colorbar的设置
    cbar.ax.set_ylabel('score', rotation=-90, va="bottom",fontsize=120,fontname='Times New Roman')#colorbar标签为‘score’，纵向放置，字体大小为18，字体为新罗马字体
    #color色条本身上刻度值大小及字体设置
    # cbar.ax.yaxis.set_major_locator(MultipleLocator(1))
    # cbar.ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    cbar.ax.tick_params(labelsize=120)
    cbarlabels = cbar.ax.get_yticklabels() 
    [label.set_fontname('Times New Roman') for label in cbarlabels]
    
    bound = []
    for i, ind in enumerate(boundary):
        if (ind!=0):
            plt.axvline(i,color='black',linewidth = '3')
            bound.append(i)
    print("boundary", bound)

    plt.savefig(output_photo_path + str(sample['feature_path']).split(".")[0].split("/")[-1]+'.png', bbox_inches = 'tight')

    plt.close(figure)
    
    print("done")

def draw_feature(x, photo_path ='./photo_feature/'):


    if not os.path.exists(photo_path):
        os.makedirs(photo_path)

    # x = x.unsqueeze(0)

    # feature = x[:, :, :2000]
    feature = x


    N, C, T = feature.size()

    data = np.array(feature.clone().data.cpu()).squeeze(0)

    # data = np.around(data, 4)

    print("figure: draw_feature")
    figure=plt.figure(facecolor='w',figsize=(300, 30))
    ax=figure.add_subplot(1,1,1)
    plt.tick_params(labelsize=30)
    ax.set_yticks(range(C))
    ax.set_xticks(range(T))

    cmap=cm.coolwarm
    # cmap=cm.get_cmap('rainbow',1000)

    map=sns.heatmap(data, cmap=cmap)
    # cb=ax.colorbar(mappable=map,cax=None,ax=None,shrink=0.5) #图例
    plt.savefig(photo_path + str(x.shape)+'.png')

    print("done")


def draw_feature_eval(x, boundary, photo_path ='./photo_feature/'):
    
    print("figure")

    if not os.path.exists(photo_path):
        os.makedirs(photo_path)
    
    boundary = np.array(boundary.clone().data.cpu()).squeeze()
    boundary =  np.where(boundary == 1)[0]
    print(boundary)
    feature = x[:, 100:150, (boundary[2]-150): (boundary[2]+150)]

    N, C, T = feature.size()

    data = np.array(feature.clone().data.cpu()).squeeze()




    # for i in range(x.shape[2]):
    #     if(boundary[i]==1):
    #         plt.axhline(i)
    #         print(i)


    # data = np.around(data, 4)
    
    figure=plt.figure(facecolor='w',figsize=(60, 10),dpi=300)
    ax=figure.add_subplot(1,1,1)
    # plt.tick_params(labelsize=30)
    ax.set_yticks(range(C))
  
    ax.set_xticks(range(T))

    cmap=cm.coolwarm
    # cmap=cm.get_cmap('rainbow',1000)

    map=sns.heatmap(data, cmap=cmap)
    # cb=ax.colorbar(mappable=map,cax=None,ax=None,shrink=0.5) #图例
    plt.savefig(photo_path + str(x.shape)+'.png')

    print("done")


def draw_boundary(b, boundary, photo_path ='./photo_boundary/'):

    if not os.path.exists(photo_path):
        os.makedirs(photo_path)

    b = b.data.cpu().squeeze().numpy()

    boundary = boundary.data.cpu().squeeze().numpy()

    fig = plt.figure(figsize=(30,30), dpi=300)#设置画布的尺寸
    ax = fig.add_subplot(111)
    ax.set_title(photo_path,fontsize= 30) # title of plot

    plt.plot(np.arange(len(boundary)), list(boundary), 'r:', label="pred_boundary")
    plt.plot(np.arange(len(b)), list(b), 'y:', label="gt_boundary")
    plt.plot(np.arange(len(b-boundary)), list(b-boundary-1), label="gt_boundary-pred_boundary")

    ax.legend(loc='upper left', fontsize=30)
    plt.savefig(photo_path + str(boundary.shape)+'.png')

