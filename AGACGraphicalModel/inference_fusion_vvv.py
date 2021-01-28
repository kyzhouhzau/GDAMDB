#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:BioNLP-HZAU Kaiyin Zhou
Reference:
[1] https://github.com/kzhai/PyLDA
[2] https://github.com/blei-lab/onlineldavb
[3] Blei D M, Ng A, Jordan M. Latent Dirichlet allocation Journal of Machine Learning Research (3)[J]. 2003.
[4] Dai M, Ming J, Cai M, et al. IGESS: a statistical approach to integrating individual-level genotype data and summary
    statistics in genome-wide association studies[J]. Bioinformatics, 2017, 33(18): 2882-2889.
"""
import numpy as np
import scipy
from scipy import special
from scipy.stats import dirichlet,binom
from scipy.stats import multinomial
from scipy import misc
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

def build_geneid_symbol(file="9606gene.txt"):
    hashdir = {}
    with open(file) as rf:
        for line in rf:
            #TODO
            contents = line.strip().split()
            taxid = contents[0]
            Geneid = contents[3]
            symbol = contents[6]
            if taxid=="9606":
                hashdir[Geneid] = symbol
    return hashdir

# from generate_test import data
# theta ~ Dir(alpha), computes E[log(theta)] given alpha.
def compute_dirichlet_expectaion(dirichlet_parameter):
    if len(dirichlet_parameter.shape)==1:
        return special.psi(dirichlet_parameter) -special.psi(np.sum(dirichlet_parameter))
    return special.psi(dirichlet_parameter)-special.psi(np.sum(dirichlet_parameter,1))[:,np.newaxis]

def build_genes_drugs(inputfile):
    genes = []
    diseases = []
    with open(inputfile,'r') as rf:
        for diseaseline in rf:
            contents = diseaseline.strip().split('\t')
            disease = contents[0]
            diseases.append(disease)
            for items in contents[1:]:
                cons = items.split('&')
                genes.append(cons[0])
    genes = sorted(set(genes))
    diseases = sorted(set(diseases))
    gene2id = {g:i for i,g in enumerate(genes)}
    disease2id = {d:i for i,d in enumerate(diseases)}
    return genes,diseases,gene2id,disease2id

def get_test(outputfolder):
    testgenes = []
    "get test from all target"
    tests = open("./{}/target_buxianzhu.txt".format(outputfolder))
    for line in tests:
        contents = line.split('\t')
        testgenes.append(contents[0])
    return testgenes

def get_data(inputfile,gene2id,disease2id,initlambda,threshold,times,outputfolder):
    """
    function1:LOF
    function2:GOF
    function3:COMPLEX
    function4:UNKNOWN
    Disease gene1&function1$P_value  gene2$function2$P_value gene3&function3$P_value gene4&function4$P_value
    :param inputfile:
    :return:geneids: Disease level gene id(Gene list); gene_cts: Disease level gene count, functions
    """
    targets={"324","4292","7048","4436","2956","7157","4595","8313","4913","4089","5378","5395","5424","5426"}
    gene_p_map = {}
    diseaseids = []
    geneids = []
    functionids = []
    p_values=[]
    jibingxianzhu=[]
    jibingbuxianzhu=[]
    jibingbuxianzhup=[]
    jibingourtarget=[]
    jibingxianzhup=[]
    jibingtests = []
    testgenes = get_test(outputfolder)
    wf = open("{}/needed_{:.0f}_{:.0f}_{:.0f}.txt".format(outputfolder,initlambda,threshold,times),'w')
    with open(inputfile,'r') as rf:
        for diseaseline in rf:
            d_genes=[]
            d_function=[]
            d_p_v=[]
            jiluxianzhu = []
            jilubuxianzhu = []
            jilupzhi=[]
            tests = []
            ourtarget = {}
            contents = diseaseline.strip().split('\t')
            count=0
            xianzhup=[]
            for i,gene_function_P in enumerate(contents[1:]):
                gene,function,P_v = gene_function_P.split("&")
                d_genes.append(gene2id[gene])
                d_function.append(int(function))
                d_p_v.append(float(P_v))
                if gene in targets:
                    ourtarget[gene]=i
                if gene in testgenes:
                    tests.append(i)
                # if float(P_v)< threshold/snpnum and int(function)!=-1:
                if float(P_v)< threshold and int(function)!=-1:
                    jiluxianzhu.append(i)
                    f=None
                    if function=="0":
                        f="LOF"
                        f="HAS"
                    elif function=="1":
                        f="GOF"
                        f="HAS"
                    elif function=="2":
                        f = "COM"
                        f = "HAS"
                    elif function=="4":
                        f="REG"
                        
                    wf.write("{}\t{}\t{}\n".format(gene,f,P_v))
                    xianzhup.append(float(P_v))
                else:
                    count+=1
                    if count<=15:
                        jilupzhi.append(float(P_v))    
                        jilubuxianzhu.append(i)
            geneids.append(d_genes)
            functionids.append(d_function)
            p_values.append(d_p_v)
            jibingxianzhu.append(jiluxianzhu)
            jibingbuxianzhu.append(jilubuxianzhu)
            jibingbuxianzhup.append(jilupzhi)
            jibingxianzhup.append(xianzhup)
            jibingourtarget.append(ourtarget)
            jibingtests.append(tests)
            diseaseids.append(disease2id[contents[0]])
    return diseaseids,geneids, functionids, p_values,jibingxianzhu,jibingbuxianzhu,jibingbuxianzhup,jibingxianzhup,jibingourtarget,jibingtests

class VariationalInference(object):
    """ all data are mapped to id mode.
    disease [d1,d2,d3,d4,d5,......]
    genes [g1,g2,g3,g4,g5,......]
    function [f1,f2,f3,f4]
    """
    def __init__(self, genes, diseases, function_ids, P_v, num_hidden_factors,initlambda=100):
        self.num_genes = len(genes)
        self.genes = genes
        self.num_diseases = len(diseases)
        self.diseases = diseases
        self.num_function = 2
        self.function_ids = function_ids#D*G
        self.num_hidden_factors = num_hidden_factors
        self.P_v = P_v
        self.initlambda = initlambda
        self._initialize()
        
    def _initialize(self):
        # self._alpha = np.ones(self.num_hidden_factors)/self.num_hidden_factors#1*K
        self._alpha = np.ones((self.num_diseases,self.num_hidden_factors))/self.num_hidden_factors#1*K
        self._pi = np.ones((self.num_diseases,self.num_hidden_factors,self.num_function))/self.num_function# 1*4
        # self._pi = np.ones((self.num_diseases,self.num_hidden_factors, self.num_function))/2# 1*4
        # self._lambda = np.ones((self.num_diseases,1))/self.initlambda#1
        self._lambda = np.ones((self.num_diseases,self.num_genes))/self.initlambda
        self._ad = np.ones((self.num_diseases,1))/2#D*1

        self._tilde_pi = np.random.gamma(100.,1./100.,(self.num_diseases,self.num_hidden_factors,self.num_function)) # K * 4
        self._tilde_alpha = np.random.gamma(100.,1./100.,(self.num_diseases,self.num_hidden_factors))#D*K
        # D*K 等同于LDA中的gamma
        self._tidle_theta = np.random.gamma(100.,1./100.,(self.num_diseases,self.num_genes,self.num_hidden_factors)) # G*K 等同于LDA中的phi
        self._tidle_lambda = np.ones((self.num_diseases,self.num_genes))/100 # D*G

    def convert_to_one_hot(self,index):
        if index==-1:
            function_id_vector = np.zeros(self.num_function)
        else:
            function_id_vector = np.zeros(self.num_function)
            # function_id_vector[index] = 1
            function_id_vector[index] = 1
        return function_id_vector

    def e_step(self,did, gene_ids, function_ids):
        # tilde_alpha, tilde_pi, tilde_lambda, tilde_theta
        Elogbeta = compute_dirichlet_expectaion(self._tilde_pi[did]) #K * 4
        Elogtheta = compute_dirichlet_expectaion(self._tilde_alpha[did]) #D * K
        # TODO E[Z]
        # for _tidle_alpha, _tidle_pi, _tidle_theta
        sum_tidle_theta=0
        sum_tidle_theta_f=0
        for gid in gene_ids[did]:
            function_id = function_ids[did][gid]
            function_id_vector=self.convert_to_one_hot(function_id)
            log_tidle_theta = Elogbeta[:,function_id].T+Elogtheta# 1*K
            # Normalize
            log_tidle_theta -= misc.logsumexp(log_tidle_theta) # 1*k
            sum_tidle_theta += np.exp(log_tidle_theta) # 1*4
            sum_tidle_theta_f += np.dot(np.exp(log_tidle_theta)[:,np.newaxis],  function_id_vector[np.newaxis,:])
        tidle_alpha_updata = self._alpha[did] + sum_tidle_theta# K*4

        # tidle_pi_updata = np.tile(self._pi[did],(1,1))+sum_tidle_theta_f # K*4
        tidle_pi_updata = self._pi[did]+sum_tidle_theta_f # K*4

        w = np.log(self._lambda[did] / (1 - self._lambda[did])+1e-6) +(self._ad[did] - 1)*np.log(self._ad[did] * self.P_v[did]+1e-6)  # G*1
       
        lambda_update = special.expit(w)  # G*1
        # updata variational paramaters
        self._tidle_lambda[did] = lambda_update
        self._tilde_alpha[did] = tidle_alpha_updata
        self._tilde_pi[did] = tidle_pi_updata


    def m_step(self,did,newton_thresh=1e-5,max_iter=1000):
        self.update_alpha(did,newton_thresh,max_iter)
        self.update_pi(did,newton_thresh,max_iter)
        self.update_a_d(self.P_v,did)
        self.update_lambda(did)

    def update_lambda(self,did):
        self._lambda[did] = self._tidle_lambda[did]

    def update_a_d(self,P_dg,did):
        self._ad[did] = -np.sum(self._tidle_lambda[did,:],0)/np.dot(self._tidle_lambda[did,:],np.log(P_dg[did]))

    def update_alpha(self,did,newton_thresh=1e-5,max_iter=1000):
        Elogtheta = compute_dirichlet_expectaion(self._tilde_alpha)
        for i in range(max_iter):
            alpha = self._alpha[did]
            L_alpha = self.compute_L_alpha(alpha,Elogtheta,did)
            DL_alpha = self.compuate_dL_alpha(alpha,Elogtheta,did)
            D2L_alpha = self.compuate_d2L_alpha(alpha)
            step_size = np.dot(np.linalg.inv(D2L_alpha), DL_alpha)
            alpha_update = self._alpha[did] - step_size
            meanchange  = np.mean(abs(alpha_update-alpha))
            # print("Alpha iteration :{} meanchange:{}".format(i,meanchange))
            self._alpha[did] = alpha_update
            if meanchange<= newton_thresh:
                break

    def update_pi(self,did,newton_thresh=1e-5,max_iter=1000):
        pis = self._pi[did]
        for k,pi in enumerate(pis):
            Elogbeta = compute_dirichlet_expectaion(self._tilde_pi[did][k])
            for i in range(max_iter):
                L_pi = self.compute_L_pi(pi,Elogbeta)
                DL_pi = self.compute_dL_pi(pi,Elogbeta)
                D2L_pi = self.compute_d2L_pi(pi)
                step_size = np.dot(np.linalg.inv(D2L_pi),DL_pi)
                pi_update = self._pi[did][k] - step_size
                meanchange = np.mean(abs(pi_update -pi))
                self._pi[did][k] = pi_update
                # print("Pi iteration :{} meanchange :{}".format(i,meanchange))
                if meanchange<= newton_thresh:
                    break
    def judge(self,A):
        B = np.linalg.eigvals(A)
        if np.all(B>0):
            return 1
        else:
            return 0
    def compute_L_alpha(self,alpha,E_ln_theta,did):
        return  special.gammaln(np.sum(alpha))-np.sum(special.gammaln(alpha))+ \
                np.dot((alpha-1)[np.newaxis,:],E_ln_theta[did,:])

    def compuate_dL_alpha(self,alpha,E_ln_theta,did):
        return special.psi(np.sum(alpha))-special.psi(alpha)+E_ln_theta[did,:]

    def compuate_d2L_alpha(self,alpha):
        c = special.polygamma(1, np.sum(alpha)) - special.polygamma(1, alpha)
        z = special.polygamma(1, np.sum(alpha))
        h = np.diag(c)
        hession=h + z - np.diag(z * np.ones_like(c))
        return hession

    def compute_L_pi(self,pi,E_ln_beta):
        return self.num_hidden_factors * (special.gammaln(np.sum(pi)) - np.sum(special.gammaln(pi))) + \
               np.sum(np.dot((pi - 1)[np.newaxis,:],E_ln_beta.T))

    def compute_dL_pi(self,pi,E_ln_beta):
        return special.psi(np.sum(pi))-special.psi(pi)+E_ln_beta

    def compute_d2L_pi(self,pi):
        c = special.polygamma(1,np.sum(pi))-special.polygamma(1,pi)
        z = special.polygamma(1,np.sum(pi))
        h = np.diag(c)
        hession = h + z - np.diag(z * np.ones_like(c))
        return hession

    def document_elbo(self,did,function_ids,gene_ids):
        ep=1e-8
        Elogtheta = compute_dirichlet_expectaion(self._tilde_alpha[did,:]) # 1*K
        Elogbeta = compute_dirichlet_expectaion(self._tilde_pi[did]) # K * 4
        Elambda = self._tidle_lambda[did,:] #D * G
        Ez = self._tidle_theta[did,:] # G*K

        E_tidle_F = np.sum([np.dot(np.dot(self.convert_to_one_hot(function_ids[did][gid]),Elogbeta.T),Ez[gid].T) for gid in gene_ids[did]])

        E_P = np.dot(Elambda,(self._ad[did]-1)*np.log(self._ad[did]*self.P_v[did]))

        E_theta = special.gammaln(np.sum(self._alpha[did]))-np.sum(special.gammaln(self._alpha[did]))+ np.dot((self._alpha[did]-1),Elogtheta)

        E_beta = self.num_hidden_factors * (special.gammaln(np.sum(self._pi[did])) - np.sum(special.gammaln(self._pi[did]))) + \
               np.sum(np.dot((self._pi[did] - 1),Elogbeta.T))

        E_gamma = np.sum(Elambda*np.log(self._lambda[did]+ep)+(1-Elambda)*np.log(1-self._lambda[did]+ep))

        E_Z = np.sum(np.dot(Ez,Elogtheta.T))

        E_q_theta = special.gammaln(np.sum(self._tilde_alpha[did,:]))-np.sum(special.gammaln(self._tilde_alpha[did,:]))+ \
                np.sum(np.dot((self._tilde_alpha[did,:]-1),Elogtheta))

        E_q_beta = special.gammaln(np.sum(self._tilde_pi[did]))-np.sum([special.gammaln(self._tilde_pi[did][k]) for k in range(self.num_hidden_factors)])+ \
                np.sum([(self._tilde_pi[did][k]-1)*Elogbeta[k].T for k in range(self.num_hidden_factors)])

        Elambda = np.array(Elambda)
        E_q_gamma = np.sum(Elambda*np.log(Elambda+ep)+(1-Elambda)*np.log(1-Elambda+ep))
    
        E_q_z = np.sum([Ez[gid]*np.log(Ez[gid].T) for gid in gene_ids[did]])
        
        document_elbo = E_tidle_F+E_P+E_theta+E_beta+E_gamma+E_Z-E_q_theta-E_q_beta-E_q_gamma-E_q_z
        return document_elbo

def inference(genes, diseases, gene2id,gene_ids, disease_ids, function_ids, p_values,jibingxianzhu,
            jibingbuxianzhu,jibingbuxianzhup,jibingxianzhup,jibingourtarget,jibingtests,initlambda,threshold,
            times,filter_count,rounders,num_hidden_factors,outputfolder,epoch):
    id2symbol = build_geneid_symbol(file="9606gene.txt")
    VI = VariationalInference(genes, diseases, function_ids, p_values, num_hidden_factors,initlambda)
    id2gene={value:key for key, value in gene2id.items()}
    # id2function={0:"REG",1:"LOF",2:"GOF",3:"COM"}
    id2function={0:"NA",1:"HAS",2:"REG"}
    for did in disease_ids:
        elboold = -2e10
        print("###################Disease number {}###################\n".format(did))
        for t in range(epoch):
            VI.e_step(did, gene_ids, function_ids)
            VI.m_step(did, newton_thresh=1e-5, max_iter=1000)
            elbo = VI.document_elbo(did, function_ids,gene_ids)
            notables=[]
            for i,item in enumerate(VI._lambda[did]):
                if i in jibingxianzhu[did]:
                    notables.append(item)
            unnotables=[] 
            for ii,iitem in enumerate(VI._lambda[did]):
                if ii in jibingbuxianzhu[did]:
                    unnotables.append(iitem)
            
            print("\nApproximate one, Model parameters:\n Lambda (1*1) lambda {}".format(list(zip(notables,jibingxianzhup[did]))))
            print("\nApproximate zero, Model parameters:\n Lambda (1*1) (lambda,Pvalue) {}".format(list(zip(unnotables,jibingbuxianzhup[did]))))
            if elbo<elboold or (elbo-elboold)/abs(elboold)<=1e-10:
            # if abs(elbo-elboold)<=0:
                break
            # if abs(elbo-elboold)<=1e-10:
            #     break
            print("(New_Elbo-old_Elbo)/abs(old_elbo)={}\n".format((elbo-elboold)/abs(elboold)))
            elboold = elbo
            print("Time {} For each elbo {} of one disease\n".format(t,elbo))
        # TODO Export 
        id2gene={value:key for key, value in gene2id.items()}
        # id2function={0:"REG",1:"LOF",2:"GOF",3:"COM"}
        # id2function={0:"NA",1:"HAS",2:"REG"}

        alllambda = VI._lambda[did].tolist()
        p_value = p_values[did]
        # targetlambda = [alllambda[int(index)] for gene,index in jibingourtarget[did].items()]
        # print(jibingxianzhu[did])
        targetlambda = [alllambda[int(index)] for index in jibingxianzhu[did]]
        print("##############目标显著################")
 
        for index in jibingxianzhu[did]:
            gene = id2gene[index]
            lambda_ = alllambda[int(index)]
            p = p_value[int(index)]

            print("gene:{}\tlambda:{} p_value:{}".format(gene,lambda_,p))
        print("##############随机采样################")
        for index in range(0,1000,100):
            gene = genes[int(index)]
            lambda_ = alllambda[int(index)]
            p = p_value[int(index)]
            print("gene:{}\tlambda:{} p_value:{}".format(gene,lambda_,p))
        p_value = Wilcoxon(targetlambda,alllambda)
        print("the lambda value of wilcoxon rank sum between target gene lambda and all lambdais {}".format(p_value))
        # print(p_values[did])
        pp_value = Wilcoxon(jibingxianzhup[did],p_values[did])
        print("the p value of wilcoxon rank sum between target gene p_value and all p_value {}".format(pp_value))
        # print("threshold :{}".format(threshold/snpnum))
        print("threshold :{}".format(threshold))
        # here i will draw some picture
        pvalue = p_values[did]
        # #11个阈值以内的p值
        specific_index=[index for index in jibingxianzhu[did]]# the index in orgin data
        test_index = [index for index in jibingtests[did]]
        pindex2pvaule={index:value for index,value in enumerate(pvalue)}
        pindex2pvaule = {index:-np.log(value) for index,value in pindex2pvaule.items()}
        orderpindex2pvalue = sorted(pindex2pvaule.items(),key=lambda X:X[1])
        orgin = [item[0] for item in orderpindex2pvalue]
        index2orgin = {index:item for index,item in enumerate(orgin)}
        origin2index = {item:index for index,item in index2orgin.items()}
        # linex = [origin2index[item[0]] for item in orderpindex2pvalue]
        linex = [index for index in range(len(orderpindex2pvalue))]
        liney = [item[1] for item in orderpindex2pvalue]
        scatter_x = [origin2index[index] for index in specific_index]
        scatter_y = [pindex2pvaule[index] for index in specific_index]
        tests_x = [origin2index[index] for index in test_index]
        test_y = [pindex2pvaule[index] for index in test_index]
        '''
        print(scatter_y)
        plt.style.use('ggplot')  # 设置绘图风格
        fig = plt.figure(figsize = (10,6))  # 设置图框的大小
        ax1 = fig.add_subplot(2,1,1)
        ax1.scatter(linex,liney,c="black",s = 3) # 绘制折线图
        ax1.scatter(scatter_x,scatter_y,marker = 'o',s = 10)
        print("一共有测试数据：{}个".format(len(tests_x)))
        ax1.scatter(tests_x,test_y,marker = 'o',c="green",s=10)
        # plt.title('Gene-P-value',color = "black",fontsize = 18)
        # plt.xlabel('Gene Index')
       # plt.ylabel('-logP')
        #lambda
        '''
        maplinex2orgin = [item[0] for item in orderpindex2pvalue]
        lambda_ = VI._lambda[did]

        lambdax = linex
        lambday = [lambda_[index] for index in maplinex2orgin]
        '''
        ax2 = fig.add_subplot(2,1,2)
        ax2.scatter(lambdax,lambday,c="black",s = 3)

        plt.xlabel('Genes ranked in ascending order of -logp')
        plt.ylabel('lambda')
        plt.savefig("picture/picture1.png",dpi = 500)
        '''        
        T_for_allround = export(did,gene_ids,p_values,VI._pi,VI._alpha,VI._lambda,
                            id2gene,id2function,initlambda,threshold,times,filter_count,rounders,outputfolder)

def export(did,genesids,p_values,pi,alpha,lambdas,id2gene,id2function,initlambda,
            threshold,times,filter_count,rounders,outputfolder):
    xianzhulambda=[]
    # wf = open("{}/{:.0f}_{:.0f}_{:.0f}.txt".format(outputfolder,initlambda,threshold,times),'w')
    w = open("{}/有三元组{:.0f}_{:.0f}_{:.0f}.txt".format(outputfolder,initlambda,threshold,times),'w')
    wx = open("{}/无三元组{:.0f}_{:.0f}_{:.0f}.txt".format(outputfolder,initlambda,threshold,times),'w')
    rounders = rounders
    T_for_allround = {}
    F_for_allround = {}
    for r in range(rounders):
        for gene,p,lambd  in zip(genesids[did],p_values[did],lambdas[did]):
            gamma = stats.bernoulli(lambd).rvs(1)[0]
            if gamma==1:
                theta = dirichlet.rvs(alpha[did], size=1, random_state=1)
                z = np.argmax(multinomial(n=1,p=theta[did]).rvs(1))
                beta = dirichlet.rvs(pi[did][z],size=1,random_state=1)
                f = np.argmax(multinomial(n=1,p=beta[did]).rvs(1))+1
            else:
                f=0
            function=id2function[f]
            if function!="NA":
                xianzhulambda.append(lambd)
                triples = (id2gene[gene],function,p)
                if triples not in T_for_allround:
                    T_for_allround[triples]=0
                else:
                    T_for_allround[triples]+=1
            else:
                triples = (id2gene[gene],function,p)
                if triples not in F_for_allround:
                    F_for_allround[triples]=0
                else:
                    F_for_allround[triples]+=1
            # wf.write("000000\tp_value\t{}\tSymbol\t{}\tCC\n".format(id2gene[gene],function))
    for tw,count in T_for_allround.items():
        if count>=filter_count:
            gene,function,p = tw
            w.write("{}\t{}\t{}\t{}\n".format(gene,function,p,count))
    for fw,count in F_for_allround.items():
        if count>=filter_count:
            gene,function,p = fw
            wx.write("{}\t{}\t{}\t{}\n".format(gene,function,p,count))
    p_value = Wilcoxon(xianzhulambda,lambdas[did].tolist())
    print("the p value of wilcoxon rank sum between predict gene lambda and all lambda is {}".format(p_value))
    return T_for_allround

def drawven(lis:list,picturename):
    plt.figure(figsize=(4, 4))
    venn2(subsets=(lis), set_labels=('Predicts', 'Targets'),set_colors=('r','g'))
    plt.savefig(picturename)

def calculatescore(inptutneed,inputpredict,inputtest):
    predicts = []
    needs = []
    tests = []
    with open(inputpredict) as rf:
        for line in rf:
            contents = line.split("\t")
            gene = contents[0]
            function = contents[1]
            # predicts.append([gene,function])
            predicts.append([gene])
    with open(inptutneed) as rf:
        for line in rf:
            contents = line.split("\t")
            gene = contents[0]
            function = contents[1]
            # needs.append([gene,function])
            needs.append([gene])
    with open(inputtest) as rf:
        for line in rf:
            contents = line.split("\t")
            gene = contents[0]
            function = contents[1]
            # tests.append([gene,function])
            tests.append([gene])
    count=0
    for item in predicts:
        if item in needs:
            # print("gene in train:{}\n".format(item))
            count+=1
    count1 = 0
    for item in predicts:
        if item in tests:
            # print("gene in tests:{}\n".format(item))
            count1+=1
    #drawven([len(predicts)-count,len(needs)-count,count],"picture/Venn1.png")# 内部评价
    #drawven([len(predicts)-count1,len(tests)-count1,count1],"picture/Venn2.png")# 内部评价
    print("有{}个重现了训练集".format(count))
    print("Train--The precision is:{}".format(count/len(predicts)))
    print("Train--The recall is:{}".format(count/len(needs))) 
    print("有{}个在测试集中".format(count1))
    print("Test--The precision is:{}".format(count1/len(predicts)))
    print("Test--The recall is:{}".format(count1/len(tests))) 
    print("needs:{}\tpredicts:{}".format(len(needs),len(predicts)))

def Wilcoxon(targetlambda,alllambda):
    res = stats.ranksums(targetlambda,alllambda)
    p_value = res.pvalue
    return p_value

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Use disease name')
    parser.add_argument('--initlambda', type=float, default = 100,required=True)
    parser.add_argument('--threshold', type=float, default = None,required=True)
    parser.add_argument('--times', type=float, default = None,required=True)
    parser.add_argument('--filter_count', type=int, default = None,required=True)
    parser.add_argument('--rounders', type=int, default = None,required=True)
    parser.add_argument('--hidden_factors', type=int, default = None,required=True)
    parser.add_argument('--inputfile', type=str, default = None, required=True)
    parser.add_argument('--outputfolder', type=str, default = None, required=True)
    parser.add_argument('--epoch', type=int, default = 1)

    args = parser.parse_args()
    initlambda = args.initlambda
    epoch = args.epoch
    threshold = args.threshold
    times = args.times
    hidden_factors = args.hidden_factors
    filter_count = args.filter_count
    rounders = args.rounders
    inputfile = args.inputfile
    outputfolder = args.outputfolder
    #filename = "data/sorted_Autsms.txt"
    #filename = "data/sorted_UKBiobank.txt"
    #filename = "data/sorted_Kunkle.txt"
    #filefolder = "Kunkle/"
    #filename = "data/sorted_IGAP.txt"
    # filename = "data/Colorectal_Cancer.txt"
    # filename = "PP.csv"
    genes,diseases,gene2id,disease2id = build_genes_drugs(inputfile)
    diseaseids,geneids, functionids, p_values,jibingxianzhu,jibingbuxianzhu, \
    jibingbuxianzhup,jibingxianzhup,jibingourtarget,jibingtests = get_data(inputfile, gene2id, disease2id,initlambda,threshold,times,outputfolder)
    inference(genes, diseases,gene2id, geneids, diseaseids, functionids,
              p_values,jibingxianzhu,jibingbuxianzhu,jibingbuxianzhup,jibingxianzhup,
               jibingourtarget,jibingtests,initlambda,threshold,times,filter_count,rounders,
               num_hidden_factors=hidden_factors,outputfolder=outputfolder,epoch=epoch)

    calculatescore("{}/needed_{:.0f}_{:.0f}_{:.0f}.txt".format(outputfolder,initlambda,threshold,times),
                    "{}/有三元组{:.0f}_{:.0f}_{:.0f}.txt".format(outputfolder,initlambda,threshold,times),
                    "{}/target_buxianzhu.txt".format(outputfolder))

if __name__=="__main__":
    main()
