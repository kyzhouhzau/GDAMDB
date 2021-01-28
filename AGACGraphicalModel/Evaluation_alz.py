# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:23:35 2019

@author: 50502
"""
from glob import glob
import os

def GWAS(GWASfile):
    with open(GWASfile) as rf:
        GWAS_gene = [line.strip().split('\t')[1] for line in rf if line.strip().split('\t')[1]!='-']
    return GWAS_gene

def Jugement(prediction_file,standard_list):
    with open(prediction_file) as rf:
        predicted_gene = [line.strip().split('\t')[0] for line in rf]
        correct_gene = [gene for gene in predicted_gene if gene in standard_list]
        if len(predicted_gene) == 0:
            percent = 0
        else:
            percent = float(len(correct_gene)/len(predicted_gene))
    return ['{}/{}'.format(len(correct_gene),len(predicted_gene)),percent]

def evaluation(filefolder,Gwaspath):
    gwas = GWAS(Gwaspath)
    needed_folder = os.path.join(filefolder,'needed*.txt')
    prediction_folder = os.path.join(filefolder,'有三元组*.txt')
    neededfilelist = glob(needed_folder)
    predictionfilelist = glob(prediction_folder)
    needed = [Jugement(file,gwas)[0] for file in neededfilelist]
    needed_percent = [Jugement(file,gwas)[1] for file in neededfilelist]
    prediction = [Jugement(file,gwas)[0] for file in predictionfilelist]
    prediction_percent = [Jugement(file,gwas)[1] for file in predictionfilelist]
    hds = [os.path.basename(file).split('.')[0].split('_')[3] for file in neededfilelist]
    lds = [os.path.basename(file).split('.')[0].split('_')[1] for file in neededfilelist]
    for hd,ld,nee,nee_p,pred,pred_p in zip(hds,lds,needed,needed_percent,prediction,prediction_percent):
        print('---------------lambda:{}--hidden factor:{}-------------------\n'.format(ld,hd))
        print('needed: {}({}) \t predited: {}({}) \n'.format(nee,nee_p,pred,pred_p))

def main():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('-d','--filefolder', type=str,required=True)
        parser.add_argument('-g','--Gwaspath', type=str,default="./generate/Whole_GWAS_gene_id.txt")
        args = parser.parse_args()
        filefolder  = args.filefolder
        Gwaspath = args.Gwaspath
        evaluation(filefolder,Gwaspath)
        
        result = Jugement('generate/IGAP_Wilcoxon/通过阈值划分.txt',GWAS(Gwaspath))
        print('-------通过阈值划分--------\n')
        print('{}\t{}'.format(result[0],result[1]))
if __name__=="__main__":
        main()
