#! -*- encoding:utf-8 -*-
from scipy import stats
import numpy as np
import argparse
import matplotlib.pyplot as plt
from colour import Color
def get_all_p(gwas_pfile):
    gene_p={}
    with open(gwas_pfile) as rf:
        for line in rf:
            contents = line.strip().split()
            for content in contents[1:]:
                ccs = content.split('&')
                gene = ccs[0]
                p_value = ccs[-1]
                gene_p[gene] = float(p_value)
    print("GWAS 包含{}个基因。".format(len(gene_p)))
    return gene_p

def get_target_p(functionchange_file,gene_p):
    fgene_p = {}
    nset = set()
    with open(functionchange_file) as rf:
        for line in rf:
            contents = line.split('\t')
            gene = contents[0]
            p_value = gene_p.get(gene,-10)
            if p_value!=-10:
                fgene_p[gene] = float(p_value)
            else:
                nset.add(gene)
    print("有{}个基因不在GWAS数据中。".format(len(nset)))
    print("有{}个基因在GWAS数据中。".format(len(fgene_p)))
    # print(count)
    return fgene_p

def Wilcoxon(targetp,allp):
    res = stats.ranksums(list(targetp),list(allp))
    p_value = res.pvalue
    return p_value

def drawp_distribution(output,targetp,allp):
    p_value = Wilcoxon(targetp,allp)

def draw_Wilcoxon(minlogpx,minlogpy,output):
    maxy = max(minlogpy)
    print(minlogpx)
    print(minlogpy)
    for x,y in zip(minlogpx,minlogpy):
        if y==maxy:
            maxx =x
    print(maxx)
    print(maxy)
    fig = plt.figure(figsize = (10,6))
    ax1 = fig.add_subplot(1,1,1)

    ax1.scatter(minlogpx,minlogpy,c="black",s = 3)
    ax1.plot([maxx,maxx],[0-10,maxy+3],"--",c='red')
    # plt.title('Wilcoxon ranks sum',color = "black",fontsize = 18)
    ax1.set_xticks([0,20,40,51,60,80,100])
    plt.xlabel('Include genes')
    plt.ylabel('-logP')
    plt.savefig(output,dpi = 500)

def draw_overlap(sorted_targetp:list,fgene_p:list,sorted_fgene_p:list,minlogpx,minlogpy,output):
    def id_map(sorted_fgene_p):
        gene_index = {}
        for i,item in enumerate(sorted_fgene_p):
            geneid = item[0]
            gene_index[geneid]=i
        return gene_index
    
    gene_index = id_map(sorted_fgene_p)
    targetx = [gene_index[item[0]] for item in sorted_targetp]
    targety = [-np.log(item[1]) for item in sorted_targetp]
    
    tfx = [gene_index[item[0]] for item in fgene_p]
    tfy = [-np.log(item[1]) for item in fgene_p]
    allx = [gene_index[item[0]] for item in sorted_fgene_p]
    ally = [-np.log(item[1]) for item in sorted_fgene_p]
    # color
    c = []
    colors =[["#CC0000"]*10,["#FA0000"]*10,["#FF0909"]*10,
    ["#FF3232"]*10,["#FF4646"]*10,["#FF6464"]*10,
    ["#FF8282"]*10,["#FFA0A0"]*(len(tfx)-70)]
    [c.extend(i) for i in colors]

    # 下面的曲线
    fig = plt.figure(figsize = (15,10))
    ax1 = fig.add_subplot(2,1,1)
    ax1.scatter(allx,ally,c="black",s = 1)
    # 黑色散点
    nall_x = []
    nall_y = []
    maline = max(ally)
    flag = 0
    for i in allx:
        if i not in tfx:
            if flag%4==0:
                nall_x.append(i)
                nall_y.append(maline+2)
            flag+=1
    # ax1.scatter(nall_x,nall_y, marker='o',c="",edgecolors='black',s = 3,linewidths=1.0)
    ax1.scatter(nall_x,nall_y, marker='o',c="#93B64E",s = 2)
    # ax1.scatter(nall_x,nall_y, marker='o',s = 5)
    ############################
    ax1.scatter(tfx,tfy,marker = 'o',c=c,s = 8)
    
    tfyline = []
    maxline1 = max(tfy)
    for i in range(len(tfy)):
        tfyline.append(maxline1+4)
    
    ax1.scatter(tfx,tfyline,marker = 'o',c=c,s = 6)
    ax1.set_xlabel('Gene Index')
    ax1.set_ylabel('-logP')

    # 画第二张图
    ax2 = fig.add_subplot(2,2,3)
    maxy = max(minlogpy)
    print(minlogpx)
    print(minlogpy)
    for x,y in zip(minlogpx,minlogpy):
        if y==maxy:
            maxx =x
    print(maxx)
    print(maxy)
    ax2.scatter(minlogpx,minlogpy,c=c,s = 8)
    # ax2.plot([maxx,maxx],[0-10,maxy+3],"--",c='red')
    # plt.title('Wilcoxon ranks sum',color = "black",fontsize = 18)
    ax2.set_xticks([0,20,40,51,60,80,100])
    ax2.set_xlabel('n-increment process in the "Synchronization Filter" module')
    ax2.set_ylabel('-logP')
    ############################

    # ax1.scatter(targetx,targety,marker = 'o',c=c,s=8)
    # 画上面
    # targetyline = []
    # maxline = max(targety)
    # for i in range(len(targety)):
    #     targetyline.append(maxline+4)
    # ax1.scatter(targetx,targetyline,marker = 'o',c="red",s=6)

    # plt.title('Gene-P-value',color = "black",fontsize = 18)
    
    plt.savefig(output,dpi = 1000)

def main():   
    parser = argparse.ArgumentParser()
    parser.add_argument("-g","--gwas_pfile",type=str,required=True)
    parser.add_argument("-d","--functionchange_file",type=str,default="../data/pubtator_Alzheimer'sdisease.txt")
    parser.add_argument("-f1","--figure_Wilcoxon",type=str,required=True)
    parser.add_argument("-f2","--figure_overlap",type=str,required=True)
    args = parser.parse_args()
    gwas_pfile = args.gwas_pfile
    functionchange_file = args.functionchange_file
    figure_Wilcoxon = args.figure_Wilcoxon
    figure_overlap =args.figure_overlap
    # gwas_pfile = "../data/sorted_IGAP.txt"
    # functionchange_file = "../data/pubtator_Alzheimer'sdisease.txt"
    # figure_Wilcoxon = "Wilcoxon_IGPA.png"
    # figure_overlap = "overlap_IGPA.png"
    gene_p = get_all_p(gwas_pfile)
    fgene_p = get_target_p(functionchange_file,gene_p) # target gene-pvalue
    sorted_fgene_p = sorted(fgene_p.items(),key = lambda X:X[1]) # all gene-pvalue
    sorted_allgene_p = sorted(gene_p.items(),key = lambda X:X[1]) # all gene-pvalue
    minlogpy  = []
    minlogpx = []
    sorted_targetp = []
    minflag = 10000
    threshold = 0
    for i in range(1,len(sorted_fgene_p)+1):
        targetp = sorted_fgene_p[:i]
        
        targetp = {item[0]:item[1] for item in targetp}
        restp = {key:value for key,value in gene_p.items()-targetp.items()}
        p_value = Wilcoxon(targetp.values(),restp.values())
        minlogpy.append(-np.log(p_value))
        minlogpx.append(len(targetp))

        if minflag>p_value:
            sorted_targetp= sorted(targetp.items(),key = lambda X:X[1])
            minflag=p_value
            threshold = max(targetp.values())
    draw_overlap(sorted_targetp,sorted_fgene_p,sorted_allgene_p,minlogpx,minlogpy,figure_overlap)
    print("threshold:{}".format(threshold))
    print("开始绘制Wilcoxon随纳入的基因数变化的图！")
    draw_Wilcoxon(minlogpx,minlogpy,figure_Wilcoxon)
    
if __name__=="__main__":
    main()
